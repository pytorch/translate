import argparse
import collections
import itertools
import math
import os
import shutil
import tempfile
import time
import torch

from typing import Any, Dict, Optional, Tuple

from fairseq import criterions, data, distributed_utils, models, options, \
    progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer

from fbtranslate import average_checkpoints
from fbtranslate import data as fbtranslate_data
from fbtranslate import generate
from fbtranslate import rnn  # noqa
from fbtranslate.research.word_prediction import word_prediction_criterion  # noqa
from fbtranslate.research.word_prediction import word_prediction_model  # noqa


def get_parser_with_args():
    parser = options.get_parser('Trainer')
    options.add_dataset_args(parser, train=True, gen=True)
    options.add_distributed_training_args(parser)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)
    options.add_generation_args(parser)

    parser.add_argument(
        '--log-verbose',
        action='store_true',
        help='Whether to output more verbose logs for debugging/profiling.',
    )

    # Adds args related to training (validation and stopping criterions).
    group = parser.add_argument_group('Optimization')
    group.add_argument(
        '--subepoch-validate-interval',
        default=0,
        type=int,
        metavar='N',
        help='Calculates loss over the validation set every N batch updates. '
        'Note that validation is done at the end of every epoch regardless. '
        'A value of <= 0 disables this.',
    )
    group.add_argument(
        '--stop-time-hr',
        default=-1,
        type=int,
        metavar='N',
        help='Stops training after N hours have elapsed. '
        'A value of < 0 disables this.',
    )
    group.add_argument(
        '--stop-no-best-validate-loss',
        default=-1,
        type=int,
        metavar='N',
        help='Stops training after N validations have been run without '
        'achieving a better loss than before. Note that this is affected by '
        '--validation-interval in how frequently we run validation in the '
        'first place. A value of < 0 disables this.',
    )
    group.add_argument(
        '--stop-no-best-bleu-eval',
        default=-1,
        type=int,
        metavar='N',
        help='Stops training after N evals have been run without '
        'achieving a better BLEU score than before. Note that this is affected '
        'by --generate-bleu-eval-interval in how frequently we run BLEU eval '
        'in the first place. A value of < 0 disables this.',
    )

    # Args related to dataset.
    group = parser.add_argument_group('Dataset and data loading')
    group.add_argument(
        '--source-vocab-file',
        default='',
        metavar='FILE',
        help='Path to text file representing the fairseq Dictionary to use. '
        'If left empty, the dict is auto-generated from source training data.',
    )
    group.add_argument(
        '--source-max-vocab-size',
        default=-1,
        type=int,
        metavar='N',
        help='If a new vocab file needs to be generated, restrict it to the '
        'top N most common words. If we re-use an existing vocab file, this '
        'flag will have no effect. A value of < 0 means no max size.',
    )
    group.add_argument(
        '--target-vocab-file',
        default='',
        metavar='FILE',
        help='Path to text file representing the fairseq Dictionary to use. '
        'If left empty, the dict is auto-generated from target training data.',
    )
    group.add_argument(
        '--target-max-vocab-size',
        default=-1,
        type=int,
        metavar='N',
        help='If a new vocab file needs to be generated, restrict it to the '
        'top N most common words. If we re-use an existing vocab file, this '
        'flag will have no effect. A value of < 0 means no max size.',
    )
    group.add_argument(
        '--train-source-text-file',
        default='',
        metavar='FILE',
        help='Path to raw text file containing source training examples. '
        'This overrides what would be loaded from the data dir.',
    )
    group.add_argument(
        '--train-target-text-file',
        default='',
        metavar='FILE',
        help='Path to raw text file containing target training examples. '
        'This overrides what would be loaded from the data dir.',
    )
    group.add_argument(
        '--eval-source-text-file',
        default='',
        metavar='FILE',
        help='Path to raw text file containing source eval examples for '
        'calculating validation loss and BLEU eval scores. '
        'This overrides what would be loaded from the data dir.',
    )
    group.add_argument(
        '--eval-target-text-file',
        default='',
        metavar='FILE',
        help='Path to raw text file containing target eval examples for '
        'calculating validation loss and BLEU eval scores. '
        'This overrides what would be loaded from the data dir.',
    )
    group.add_argument(
        '--penalized-target-tokens-file',
        default='',
        metavar='FILE',
        help='Path to text file of tokens to receive a penalty in decoding.'
        'If left empty, no penalty will be applied',
    )

    # Adds args related to checkpointing.
    group = parser.add_argument_group('Checkpointing')
    group.add_argument(
        '--no-end-of-epoch-checkpoints',
        action='store_true',
        help='Disables saving checkpoints at the end of the epoch. '
        'This differs from --no-save and --no-epoch-checkpoints in that it '
        'still allows for intra-epoch checkpoints if --save-interval is set.'
    )

    # Adds args for generating intermediate BLEU eval while training.
    group = parser.add_argument_group('Generation')
    group.add_argument(
        '--generate-bleu-eval-per-epoch',
        action='store_true',
        help='Whether to generate BLEU score eval after each epoch.',
    )
    group.add_argument(
        '--generate-bleu-eval-interval',
        default=0,
        type=int,
        metavar='N',
        help='Does BLEU eval every N batch updates. Note that '
        '--save-interval also affects this - we can only eval as '
        'frequently as a checkpoint is written. A value of <= 0 '
        'disables this.',
    )
    group.add_argument(
        '--generate-bleu-eval-avg-checkpoints',
        default=1,
        type=int,
        metavar='N',
        help='Maximum number of last N checkpoints to average over when '
        'doing BLEU eval. Must be >= 1.',
    )
    group.add_argument(
        '--continuous-averaging-after-epochs',
        type=int,
        default=-1,
        help=(
            'Average parameter values after each step, beginning after the '
            'specified number of epochs. '
            'IMPORTANT NOTE: incompatible with '
            '--generate-bleu-eval-avg-checkpoints > 1'
        ),
    )
    group.add_argument(
        '--append-eos-to-source',
        action='store_true',
        default=False,
        help=(
            'Apppend EOS to source sentences (instead of just target). '
            'Note that this always in effect when using binarized data.'
        ),
    )
    group.add_argument(
        '--reverse-source',
        action='store_true',
        default=False,
        help='Feed source sentence to model in reverse order.',
    )
    group.add_argument(
        '--word-reward',
        type=float,
        default=0.0,
        help=(
            'Value to add to (log-prob) score for each token except EOS. '
            'IMPORTANT NOTE: higher values of --lenpen and --word-reward '
            'both encourage longer translations, while higher values of '
            '--unkpen penalize UNKs more.'
        ),
    )

    return parser


def parse_args_and_arch(parser):
    args = options.parse_args_and_arch(parser)
    # Prevents generate from printing individual translated sentences when
    # calculating BLEU score.
    args.quiet = True

    if not(
        bool(args.train_source_text_file) ==
        bool(args.train_target_text_file) ==
        bool(args.eval_source_text_file) ==
        bool(args.eval_target_text_file)
    ):
        raise ValueError(
            'If any of --train-source-text-file, --train-target-text-file, '
            '--eval-source-text-file, or --eval-target-text-file is specified, '
            'all must be specified.'
        )
    if args.train_source_text_file and args.data:
        print(
            'Using text data files from command line flags instead of '
            'data from the directory {}'.format(args.data)
        )

    print(args)
    return args


def load_existing_checkpoint(save_dir, restore_file, trainer):
    # Load the latest checkpoint if one is available
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        print(
            '| loaded checkpoint {} (epoch {})'.format(
                checkpoint_path,
                extra_state['epoch'],
            )
        )
        # batch_offset being None denotes this was a checkpoint saved at
        # the end of an epoch (after the last batch).
        if extra_state['batch_offset'] is None:
            trainer.lr_step(extra_state['epoch'])
            extra_state['epoch'] += 1
            extra_state['batch_offset'] = 0
    else:
        extra_state = {
            'epoch': 1,
            'batch_offset': 0,
            'val_loss': None,
        }
    return extra_state


def setup_training(args):
    """Parse args, load dataset, and load model trainer."""
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = [args.train_subset, args.valid_subset]
    if (data.has_binary_files(args.data, splits) and
        not (args.train_source_text_file and
             args.train_target_text_file and
             args.eval_source_text_file and
             args.eval_target_text_file)):
        if args.log_verbose:
            print('Starting to load binary data files.', flush=True)
        dataset = data.load_dataset(
            args.data,
            splits,
            args.source_lang,
            args.target_lang,
        )
    else:
        if args.source_lang is None:
            args.source_lang = 'src'
        if args.target_lang is None:
            args.target_lang = 'tgt'

        if (args.train_source_text_file and
                args.train_target_text_file and
                args.eval_source_text_file and
                args.eval_target_text_file):
            train_corpus = fbtranslate_data.ParallelCorpusConfig(
                source=fbtranslate_data.CorpusConfig(
                    dialect=args.source_lang,
                    data_file=args.train_source_text_file,
                    vocab_file=args.source_vocab_file,
                    max_vocab_size=args.source_max_vocab_size,
                ),
                target=fbtranslate_data.CorpusConfig(
                    dialect=args.target_lang,
                    data_file=args.train_target_text_file,
                    vocab_file=args.target_vocab_file,
                    max_vocab_size=args.target_max_vocab_size,
                ),
            )
            eval_corpus = fbtranslate_data.ParallelCorpusConfig(
                source=fbtranslate_data.CorpusConfig(
                    dialect=args.source_lang,
                    data_file=args.eval_source_text_file,
                    # Vocab configs aren't relevant for the eval corpus.
                    vocab_file=None,
                    max_vocab_size=-1,
                ),
                target=fbtranslate_data.CorpusConfig(
                    dialect=args.target_lang,
                    data_file=args.eval_target_text_file,
                    # Vocab configs aren't relevant for the eval corpus.
                    vocab_file=None,
                    max_vocab_size=-1,
                ),
            )
        else:
            train_corpus, eval_corpus = fbtranslate_data.infer_file_paths(
                data_dir=args.data,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                train_split=args.train_subset,
                eval_split=args.valid_subset,
            )

        if args.log_verbose:
            print('Starting to load raw text files.', flush=True)
        dataset = fbtranslate_data.load_raw_text_dataset(
            train_corpus=train_corpus,
            eval_corpus=eval_corpus,
            train_split=args.train_subset,
            eval_split=args.valid_subset,
            save_dir=args.save_dir,
            args=args,
            penalized_target_tokens_file=args.penalized_target_tokens_file,
            append_eos_to_source=args.append_eos_to_source,
            reverse_source=args.reverse_source,
        )
    if args.log_verbose:
        print('Finished loading dataset', flush=True)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    print(
        '| [{}] dictionary: {} types'.format(
            dataset.src,
            len(dataset.src_dict),
        )
    )
    print(
        '| [{}] dictionary: {} types'.format(
            dataset.dst,
            len(dataset.dst_dict),
        )
    )
    for split in splits:
        print(
            '| {} {} {} examples'.format(
                args.data,
                split,
                len(dataset.splits[split]),
            )
        )

    # Build model and criterion
    model = models.build_model(args, dataset.src_dict, dataset.dst_dict)
    criterion = criterions.build_criterion(
        args, dataset.src_dict, dataset.dst_dict
    )
    print(
        '| model {}, criterion {}'.format(
            args.arch,
            criterion.__class__.__name__,
        )
    )
    print(
        '| num. model params: {}'.
        format(sum(p.data.numel() for p in model.parameters()))
    )

    # Build trainer
    trainer = Trainer(args, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print(
        '| max tokens per GPU = {} and max sentences per GPU = {}'.format(
            args.max_tokens,
            args.max_sentences,
        ),
        flush=True,
    )

    extra_state = load_existing_checkpoint(
        args.save_dir,
        args.restore_file,
        trainer,
    )

    return extra_state, trainer, dataset


def main(args):
    """Train the model for multiple epochs."""
    extra_state, trainer, dataset = setup_training(args)

    train_iterator = train(
        args=args,
        extra_state=extra_state,
        trainer=trainer,
        dataset=dataset,
    )

    for _ in train_iterator:
        pass


def train(
    args,
    extra_state,
    trainer,
    dataset,
):
    start_time = time.time()

    # offset for current epoch (may be different from checkpoint offset)
    starting_offset = extra_state['batch_offset']

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and extra_state['epoch'] <= max_epoch:
        """Train the model for one epoch."""

        itr, progress, extra_meters = setup_epoch(
            args=args,
            epoch=extra_state['epoch'],
            batch_offset=starting_offset,
            trainer=trainer,
            dataset=dataset,
        )

        last_bleu_eval = 0
        for i, sample in enumerate(itr, start=starting_offset):
            log_output = trainer.train_step(sample)

            train_stats = log_mid_epoch_stats(
                trainer=trainer,
                progress=progress,
                extra_meters=extra_meters,
                log_output=log_output,
            )

            if (
                args.continuous_averaging_after_epochs >= 0 and
                extra_state['epoch'] > args.continuous_averaging_after_epochs
            ):
                model_param_dict = trainer.model.state_dict()
                if 'param_totals' not in extra_state:
                    extra_state['param_totals'] = {}
                    for name, value in model_param_dict.items():
                        extra_state['param_totals'][name] = value.clone()
                else:
                    for name, value in model_param_dict.items():
                        extra_state['param_totals'][name] += value
                if 'param_accum_count' not in extra_state:
                    extra_state['param_accum_count'] = 1
                else:
                    extra_state['param_accum_count'] += 1

            if i == starting_offset:
                # ignore the first mini-batch in words-per-second calculation
                trainer.get_meter('wps').reset()

            num_updates = trainer.get_num_updates()
            do_validate = (
                args.subepoch_validate_interval > 0 and
                num_updates % args.subepoch_validate_interval == 0
            )
            do_save = (
                not args.no_save and
                args.save_interval > 0 and
                num_updates % args.save_interval == 0
            )
            do_eval_bleu = (
                # We can only do BLEU eval when we have a new checkpoint to load.
                do_save and
                args.generate_bleu_eval_interval > 0 and
                num_updates - last_bleu_eval >= args.generate_bleu_eval_interval
            )
            if do_eval_bleu:
                last_bleu_eval = num_updates

            extra_state['batch_offset'] = i + 1

            (
                _,
                val_ppl,
                val_bleu,
                stop_training_mid_epoch,
            ) = validate_save_and_evaluate_bleu(
                args=args,
                trainer=trainer,
                dataset=dataset,
                extra_state=extra_state,
                do_validate=do_validate,
                do_save=do_save,
                do_eval_bleu=do_eval_bleu,
            )
            yield (
                trainer.get_num_updates(),
                {
                    'train_ppl': train_stats['ppl'],
                    'tune_ppl': val_ppl,
                    'tune_bleu': val_bleu,
                },
            )

            if stop_training_mid_epoch:
                break

        # log end-of-epoch stats
        train_stats = log_end_epoch_stats(
            trainer=trainer,
            progress=progress,
            extra_meters=extra_meters,
        )

        if stop_training_mid_epoch:
            break

        # batch_offset being None denotes the end of an epoch.
        extra_state['batch_offset'] = None

        (
            val_loss,
            val_ppl,
            val_bleu,
            stop_training_end_of_epoch,
        ) = validate_save_and_evaluate_bleu(
            args=args,
            trainer=trainer,
            dataset=dataset,
            extra_state=extra_state,
            do_validate=True,
            do_save=not args.no_save and not args.no_end_of_epoch_checkpoints,
            do_eval_bleu=args.generate_bleu_eval_per_epoch,
        )
        extra_state['val_loss'] = val_loss
        yield (
            trainer.get_num_updates(),
            {
                'train_ppl': train_stats['ppl'],
                'tune_ppl': val_ppl,
                'tune_bleu': val_bleu,
            },
        )
        if stop_training_end_of_epoch:
            break

        lr = trainer.lr_step(extra_state['epoch'], val_loss)
        extra_state['epoch'] += 1
        starting_offset = 0

        if is_training_over_time_limit(start_time, args.stop_time_hr):
            break

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    if hasattr(evaluate_bleu, 'best') and hasattr(evaluate_bleu, 'best_epoch'):
        print(
            f'| Best BLEU score of {evaluate_bleu.best} was from '
            f'epoch {evaluate_bleu.best_epoch}'
        )


def is_training_over_time_limit(start_time, stop_time):
    elapsed_hr = (time.time() - start_time) / (60 * 60)
    training_over_time_limit = False
    if stop_time >= 0 and elapsed_hr > stop_time:
        print(
            'Stopping training due to stop time limit - it has been '
            '{} hours since starting training at {}.'.format(
                elapsed_hr, start_time
            )
        )
        training_over_time_limit = True
    return training_over_time_limit


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def setup_epoch(
    args,
    epoch,
    batch_offset,
    trainer,
    dataset,
):
    """Sets up data and progress meters for one epoch."""
    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = args.seed + epoch
    torch.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(args.max_source_positions, trainer.get_model().max_encoder_positions()),
        min(args.max_target_positions, trainer.get_model().max_decoder_positions())
    )

    # Initialize dataloader, starting at batch_offset
    itr = dataset.train_dataloader(
        args.train_subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions_train,
        seed=seed,
        epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum),
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args,
        itr,
        epoch,
        no_progress_bar='simple',
    )
    itr = itertools.islice(progress, batch_offset, None)

    # reset training meters
    for k in [
        'train_loss',
        'train_nll_loss',
        'wps',
        'ups',
        'wpb',
        'bsz',
        'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    return itr, progress, extra_meters


def log_mid_epoch_stats(
    trainer,
    progress,
    extra_meters,
    log_output,
):
    stats = get_training_stats(trainer)
    for k, v in log_output.items():
        if k in ['loss', 'nll_loss']:
            continue  # these are already logged above
        if 'loss' in k:
            extra_meters[k].update(v, log_output['sample_size'])
        else:
            extra_meters[k].update(v)
        stats[k] = extra_meters[k].avg
    progress.log(stats)
    return stats


def log_end_epoch_stats(
    trainer,
    progress,
    extra_meters,
):
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)
    return stats


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    return stats


def save_checkpoint_maybe_continuous(filename, trainer, extra_state):
    if 'param_totals' not in extra_state:
        trainer.save_checkpoint(filename, extra_state)
        return

    # trainer.save_checkpoint generates the structure used to save checkpoints
    with tempfile.TemporaryFile() as buffer:
        trainer.save_checkpoint(buffer, extra_state)
        buffer.seek(0)
        state = torch.load(
            buffer,
            map_location=(
                lambda s, _:
                    torch.serialization.default_restore_location(s, 'cpu')),
        )
        buffer.close()

    param_accum_count = extra_state['param_accum_count']
    for param_name, param_value in extra_state['param_totals'].items():
        state['model'][param_name] = param_value / param_accum_count
    torch.save(state, filename)


def save_checkpoint(trainer, args, extra_state):
    epoch = extra_state['epoch']
    batch_offset = extra_state['batch_offset']
    val_loss = extra_state['val_loss']

    if args.log_verbose:
        print(
            f'Preparing to save checkpoints for epoch {epoch}, '
            f'offset {batch_offset}.',
            flush=True
        )

    # This uses a function-local variable as basically a namescoped global
    # variable, like save_checkpoint.best below.
    if not hasattr(save_checkpoint, 'last_checkpoints'):
        if args.generate_bleu_eval_avg_checkpoints < 1:
            raise argparse.ArgumentTypeError(
                '--generate-bleu-eval-avg-checkpoints must be >= 1.'
            )
        save_checkpoint.last_checkpoints = collections.deque(
            maxlen=args.generate_bleu_eval_avg_checkpoints
        )

    # batch_offset being None means that we're at the end of an epoch.
    if batch_offset is None:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(
                args.save_dir,
                'checkpoint{}.pt'.format(epoch),
            )
            save_checkpoint_maybe_continuous(
                epoch_filename,
                trainer,
                extra_state,
            )
            save_checkpoint.last_checkpoints.append(epoch_filename)

        assert val_loss is not None
        if not hasattr(
            save_checkpoint, 'best'
        ) or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
            save_checkpoint_maybe_continuous(
                best_filename,
                trainer,
                extra_state,
            )

    # Otherwise, we're in the middle of an epoch.
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir,
            'checkpoint{}_{}.pt'.format(epoch, batch_offset),
        )
        save_checkpoint_maybe_continuous(
            epoch_filename,
            trainer,
            extra_state,
        )
        save_checkpoint.last_checkpoints.append(epoch_filename)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    save_checkpoint_maybe_continuous(
        last_filename,
        trainer,
        extra_state,
    )

    # This ensures we'll always have at least one checkpoint in the list to use
    # for BLEU eval, even if we're not saving epoch checkpoints.
    if args.no_epoch_checkpoints:
        save_checkpoint.last_checkpoints.clear()
        save_checkpoint.last_checkpoints.append(last_filename)
    if args.log_verbose:
        print(
            f'Finished saving checkpoints for epoch {epoch}, '
            f'offset {batch_offset}.',
            flush=True
        )


def validate(args, trainer, dataset, subset, epoch):
    """Evaluate the model on the validation set and return the average loss."""
    # Initialize dataloader
    max_positions_valid = (
        trainer.get_model().max_encoder_positions(),
        trainer.get_model().max_decoder_positions(),
    )
    itr = dataset.eval_dataloader(
        subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args, itr, epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )

    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    val_loss = stats['valid_loss']
    val_ppl = stats['valid_ppl']
    if not hasattr(validate, 'lowest_loss') or val_loss < validate.lowest_loss:
        validate.lowest_loss = val_loss
        validate.num_since_best = 0
    elif not hasattr(validate, 'num_since_best'):
        validate.num_since_best = 1
    else:
        validate.num_since_best += 1

    stop_due_to_val_loss = False
    if (args.stop_no_best_validate_loss >= 0 and
            validate.num_since_best > args.stop_no_best_validate_loss):
        stop_due_to_val_loss = True
        print(
            'Stopping training due to validation score stagnation - last best '
            'validation loss of {} (current loss: {}) was {} validations ago.'.
            format(
                validate.lowest_loss, val_loss, validate.num_since_best
            )
        )
    return val_loss, val_ppl, stop_due_to_val_loss


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    return stats


def _save_averaged_checkpoint(args, epoch, offset):
    if args.log_verbose:
        print(
            f'Reading {len(save_checkpoint.last_checkpoints)} previous '
            f'checkpoints for averaging in epoch {epoch}, offset {offset}.',
            flush=True
        )
    averaged_state = average_checkpoints.average_checkpoints(
        save_checkpoint.last_checkpoints)
    filename = os.path.join(
        args.save_dir, 'averaged_checkpoint{}_{}.pt'.format(epoch, offset))
    if args.log_verbose:
        print(
            f'Preparing to save averaged checkpoint for '
            f'epoch {epoch}, offset {offset}.',
            flush=True
        )
    torch.save(averaged_state, filename)
    if args.log_verbose:
        print(
            f'Finished saving averaged checkpoint for '
            f'epoch {epoch}, offset {offset}.',
            flush=True
        )
    return filename


def calculate_bleu_on_subset(args, dataset, epoch, offset, dataset_split):
    scorer, num_sentences, gen_timer = generate.generate_score(
        args=args,
        dataset=dataset,
        dataset_split=dataset_split,
    )

    print(
        f'| epoch {epoch:03d} | offset {offset} '
        f'| Eval on {dataset_split} subset '
        f'with beam={args.beam}: {scorer.result_string()}. '
        f'Generated {num_sentences} sentences ({gen_timer.n} tokens) '
        f'in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s).',
        flush=True
    )
    return scorer.score()


def evaluate_bleu(args, dataset, epoch, offset):
    filename = _save_averaged_checkpoint(args, epoch, offset)
    args.path = [filename]
    val_bleu = calculate_bleu_on_subset(
        args=args,
        dataset=dataset,
        epoch=epoch,
        offset=offset,
        dataset_split=args.valid_subset,
    )
    if (not hasattr(evaluate_bleu, 'best') or
            val_bleu > evaluate_bleu.best):
        evaluate_bleu.best = val_bleu
        evaluate_bleu.best_epoch = epoch
        best_filename = os.path.join(
            args.save_dir, 'averaged_checkpoint_best.pt')
        shutil.copy2(filename, best_filename)
        evaluate_bleu.num_since_best = 0
    elif not hasattr(evaluate_bleu, 'num_since_best'):
        evaluate_bleu.num_since_best = 1
    else:
        evaluate_bleu.num_since_best += 1

    stop_due_to_val_bleu = False
    if (args.stop_no_best_bleu_eval >= 0 and
            evaluate_bleu.num_since_best > args.stop_no_best_bleu_eval):
        stop_due_to_val_bleu = True
        print(
            'Stopping training due to BLEU score stagnation on valid set - '
            'last best BLEU score of {} (current score: {}) was {} evals ago.'
            .format(
                evaluate_bleu.best,
                val_bleu,
                evaluate_bleu.num_since_best,
            )
        )
    return val_bleu, stop_due_to_val_bleu


def validate_save_and_evaluate_bleu(
    args,
    trainer,
    dataset,
    extra_state: Dict[str, Any],
    do_validate: bool,
    do_save: bool,
    do_eval_bleu: bool,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    bool,
]:
    # evaluate on validate set
    val_loss = None
    val_ppl = None
    stop_due_to_val_loss = False
    if do_validate:
        val_loss, val_ppl, stop_due_to_val_loss = validate(
            args=args,
            trainer=trainer,
            dataset=dataset,
            subset=args.valid_subset,
            epoch=extra_state['epoch'],
        )
    extra_state['val_loss'] = val_loss

    val_bleu = None
    stop_due_to_val_bleu = False
    if do_save and distributed_utils.is_master(args):
        # save checkpoint
        save_checkpoint(
            trainer=trainer,
            args=args,
            extra_state=extra_state,
        )
        if do_eval_bleu:
            val_bleu, stop_due_to_val_bleu = evaluate_bleu(
                args=args,
                dataset=dataset,
                epoch=extra_state['epoch'],
                offset=extra_state['batch_offset'],
            )

    return (
        val_loss,
        val_ppl,
        val_bleu,
        stop_due_to_val_loss or stop_due_to_val_bleu,
    )
