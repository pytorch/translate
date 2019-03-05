#!/usr/bin/env python3

import math
from optparse import OptionParser

from pytorch_translate.research.unsupervised_morphology import unsupervised_morphology


def get_arg_parser():
    parser = OptionParser()
    parser.add_option(
        "--train-file",
        dest="train_file",
        help="Raw text as training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--model",
        dest="model_path",
        help="Path to the model file.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--iter",
        type="int",
        dest="em_iter",
        help="Number of EM training epochs.",
        default=30,
    )
    parser.add_option(
        "--num-cpu",
        type="int",
        dest="num_cpus",
        help="Number of CPUs for parallel training.",
        default=10,
    )
    parser.add_option(
        "--input",
        dest="input_file",
        help="Raw text to segment.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--output",
        dest="output_file",
        help="Segmented output file.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--save-checkpoint", action="store_true", dest="save_checkpoint", default=False
    )
    parser.add_option(
        "--smooth-const",
        type="float",
        help="Constant float value for smoothing probabilities.",
        dest="smooth_const",
        default=2,
    )
    parser.add_option(
        "--len-cost-pow",
        type="float",
        help="Power value for penalizing sequence for their length.",
        dest="len_cost_pow",
        default=2,
    )
    parser.add_option(
        "--max-morph-len",
        type="float",
        help="Max morphology length.",
        dest="max_morph_len",
        default=8,
    )
    parser.add_option(
        "--hard-em", action="store_true", dest="use_hardEM", default=False
    )
    parser.add_option(
        "--investigate",
        action="store_true",
        dest="investigate",
        help="Manually investigate param values for error analysis.",
        default=False,
    )
    parser.add_option(
        "--context",
        action="store_true",
        dest="context",
        help="Use transition probabilities.",
        default=False,
    )
    return parser


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    if options.train_file is not None and options.model_path is not None:
        model = unsupervised_morphology.UnsupervisedMorphology(
            input_file=options.train_file,
            smoothing_const=options.smooth_const,
            use_hardEM=options.use_hardEM,
            max_morph_len=options.max_morph_len,
            len_cost_pow=options.len_cost_pow,
        )
        model.expectation_maximization(
            options.train_file,
            options.em_iter,
            options.num_cpus,
            options.model_path if options.save_checkpoint else None,
        )
        if not options.save_checkpoint:
            model.params.save(options.model_path)

    if (
        options.input_file is not None
        and options.output_file is not None
        and options.model_path is not None
    ):
        model = unsupervised_morphology.MorphologyHMMParams.load(options.model_path)
        segmentor = unsupervised_morphology.MorphologySegmentor(model)
        segment_cache = {}
        writer = open(options.output_file, "w", encoding="utf-8")
        with open(options.input_file, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                writer.write(segmentor.segment_sentence(line.strip()) + "\n")
        writer.close()

    if options.investigate and options.model_path is not None:
        model = unsupervised_morphology.MorphologyHMMParams.load(options.model_path)
        segmentor = unsupervised_morphology.MorphologySegmentor(model)
        while True:
            message = " ".join(
                [
                    "input options: 1) e [str] for emission probs,",
                    "2) s [str] for segmenting sentence:\n",
                ]
            )
            input_command = input(message).strip().split()

            if len(input_command) > 1 and input_command[0] == "e":
                e = (
                    model.morph_emit_probs[input_command[1]]
                    if input_command[1] in model.morph_emit_probs
                    else 0
                )
                ln = math.pow(len(input_command[1]) - 1, model.len_cost_pow)
                cost_coef = math.exp(-ln)

                emission_prob = model.emission_prob(input_command[1])
                print(e, cost_coef, e * cost_coef)
            elif len(input_command) > 1 and input_command[0] == "s":
                segmented = segmentor.segment_sentence(" ".join(input_command[1:]))
                print(segmented)
