#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import options, progress_bar, utils
from pytorch_translate import (  # noqa  # noqa  # noqa
    hybrid_transformer_rnn,
    options as pytorch_translate_options,
    rnn,
    transformer,
    utils as pytorch_translate_utils,
)
from pytorch_translate.constants import CHECKPOINT_PATHS_DELIMITER
from pytorch_translate.tasks.pytorch_translate_multi_task import (  # noqa
    PyTorchTranslateMultiTask,
)


def compute_top_k(
    task,
    models,
    dataset,
    k,
    use_cuda,
    max_tokens=None,
    max_sentences=None,
    progress_bar_args=None,
):
    """
    This function runs forward computation on an ensemble of trained models
    using binarized parallel training data and returns the top-k probabilities
    and their corresponding token indices for each output step.

    Returns: (top_k_scores, top_k_indices)
        Each a NumPy array of size (total_target_tokens, k)
    """
    top_k_scores_list = [None for _ in range(len(dataset))]
    top_k_indices_list = [None for _ in range(len(dataset))]

    itr = task.get_batch_iterator(
        dataset=dataset, max_tokens=max_tokens, max_sentences=max_sentences
    ).next_epoch_itr(shuffle=False)
    if progress_bar_args is not None:
        itr = progress_bar.build_progress_bar(
            args=progress_bar_args,
            iterator=itr,
            prefix=f"top-k probs eval",
            no_progress_bar="simple",
        )

    for sample in itr:
        sentence_ids = sample["id"]
        target_lengths = (
            (sample["net_input"]["prev_output_tokens"] != dataset.tgt_dict.pad())
            .sum(axis=1)
            .numpy()
        )
        if use_cuda:
            sample = utils.move_to_cuda(sample)
        avg_probs = None
        for model in models:
            with torch.no_grad():
                net_output = model(**sample["net_input"])
                probs = model.get_normalized_probs(net_output, log_probs=False)
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
        avg_probs.div_(len(models))

        top_k_avg_probs, indices = torch.topk(avg_probs, k=k)

        top_k_probs_normalized = F.normalize(top_k_avg_probs, p=1, dim=2).cpu()
        indices = indices.cpu()

        for i, sentence_id in enumerate(sentence_ids):
            length = target_lengths[i]
            top_k_scores_list[sentence_id] = top_k_probs_normalized[i][:length].numpy()
            top_k_indices_list[sentence_id] = indices[i][:length].numpy()

    assert all(
        top_k_scores is not None for top_k_scores in top_k_scores_list
    ), "scores not calculated for all examples!"
    assert all(
        top_k_indices is not None for top_k_indices in top_k_indices_list
    ), "indices not calculated for all examples!"

    top_k_scores = np.concatenate(top_k_scores_list, axis=0)
    top_k_indices = np.concatenate(top_k_indices_list, axis=0)

    return top_k_scores, top_k_indices


def save_top_k(args):
    """
    This function runs forward computation on an ensemble of trained models
    using binarized parallel training data and saves the top-k probabilities
    and their corresponding token indices for each output step.

    Note that the Python binary accepts all generation params, but ignores
    inapplicable ones (such as those related to output length). --max-tokens
    is of particular importance to prevent memory errors.
    """
    pytorch_translate_options.print_args(args)
    use_cuda = torch.cuda.is_available() and not getattr(args, "cpu", False)

    (
        models,
        model_args,
        task,
    ) = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path.split(CHECKPOINT_PATHS_DELIMITER)
    )
    for model in models:
        model.eval()
        if use_cuda:
            model.cuda()

    append_eos_to_source = model_args[0].append_eos_to_source
    reverse_source = model_args[0].reverse_source
    assert all(
        a.append_eos_to_source == append_eos_to_source
        and a.reverse_source == reverse_source
        for a in model_args
    )
    assert (
        args.source_binary_file != "" and args.target_binary_file != ""
    ), "collect_top_k_probs requires binarized data."
    task.load_dataset(args.gen_subset, args.source_binary_file, args.target_binary_file)

    assert (
        args.top_k_probs_binary_file != ""
    ), "must specify output file (--top-k-probs-binary-file)!"
    output_path = args.top_k_probs_binary_file

    dataset = task.dataset(args.gen_subset)

    top_k_scores, top_k_indices = compute_top_k(
        task=task,
        models=models,
        dataset=dataset,
        k=args.k_probs_to_collect,
        use_cuda=use_cuda,
        max_tokens=args.teacher_max_tokens,
        max_sentences=args.max_sentences,
        progress_bar_args=args,
    )

    np.savez(output_path, top_k_scores=top_k_scores, top_k_indices=top_k_indices)
    print(
        f"Saved top {top_k_scores.shape[1]} probs for a total of "
        f"{top_k_scores.shape[0]} tokens to file {output_path}"
    )


def get_parser_with_args():
    parser = options.get_parser("Collect Top-K Probs", default_task="pytorch_translate")
    pytorch_translate_options.add_verbosity_args(parser)
    pytorch_translate_options.add_dataset_args(parser, gen=True)
    generation_group = options.add_generation_args(parser)

    generation_group.add_argument(
        "--source-binary-file",
        default="",
        help="Path for the binary file containing source eval examples. "
        "(Overrides --source-text-file. Must be used in conjunction with "
        "--target-binary-file).",
    )
    generation_group.add_argument(
        "--target-binary-file",
        default="",
        help="Path for the binary file containing target eval examples. "
        "(Overrides --target-text-file. Must be used in conjunction with "
        "--source-binary-file).",
    )
    generation_group.add_argument(
        "--k-probs-to-collect",
        type=int,
        default=8,
        help="Number of probabilities to collect for each output step.",
    )
    generation_group.add_argument(
        "--top-k-probs-binary-file",
        type=str,
        default="",
        help="File into which to save top-K probabilities for each token.",
    )
    generation_group.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=1000,
        help="Maximum number of words in minibatch for teacher to score.",
    )
    return parser


def main():
    parser = get_parser_with_args()
    args = options.parse_args_and_arch(parser)
    save_top_k(args)


if __name__ == "__main__":
    main()
