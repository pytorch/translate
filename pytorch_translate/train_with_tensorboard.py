#!/usr/bin/env python3

from fairseq import distributed_utils, options
from pytorch_translate import train

import tensorboard_logger

# The map from stats_to_log to their log frequency, to extend stats for logging,
# yield more stats from `train' function in `train.py`, then capture it here.
# mid_epoch : logged every step against get_num_updates
# end_epoch : logged at the end of every epoch
LOGGED_STATS_MAP = {
    "train_ppl" : 'mid_epoch',
    "tune_ppl" : 'mid_epoch',
    "tune_bleu" : 'end_epoch'
}


def single_process_tensorboard(args):
    # For multiprocess training, only the master thread needs to configure log_dir
    if args.distributed_world_size == 1 or distributed_utils.is_master(args):
        tensorboard_logger.configure(
            args.save_dir + "/runs/" + args.tensorboard_dir)

    extra_state, trainer, task, epoch_itr = train.setup_training(args)

    train_iterator = train.train(
        args=args,
        extra_state=extra_state,
        trainer=trainer,
        task=task,
        epoch_itr=epoch_itr,
    )

    for num_updates, stats in train_iterator:
        if distributed_utils.is_master(args):
            for k, v in LOGGED_STATS_MAP.items():
                if stats[k] is not None:
                    if v == 'mid_epoch':
                        tensorboard_logger.log_value(
                            k, float(stats[k]), num_updates)
                    else:
                        tensorboard_logger.log_value(
                            k, float(stats[k]), extra_state['epoch'])


if __name__ == "__main__":
    parser = train.get_parser_with_args()
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="run-1234",
        help="The log directory for tensorboard, used as `tensorboard --logdir DIR`"
    )
    args = options.parse_args_and_arch(parser)
    train.validate_and_set_default_args(args)
    print(args)
    train.main(args, single_process_tensorboard)
