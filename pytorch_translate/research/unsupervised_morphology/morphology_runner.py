#!/usr/bin/env python3

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
        "--no-affix-symbol",
        action="store_false",
        dest="add_affix_symbols",
        default=True,
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
        "--normal-init",
        action="store_true",
        help="Initialize parameters with samples from normal distribution.",
        dest="normal_init",
        default=False,
    )
    parser.add_option(
        "--normal-mean",
        type="float",
        help="Mean for the normal distribution in initialization.",
        dest="normal_mean",
        default=2,
    )
    parser.add_option(
        "--normal-stddev",
        type="float",
        help="Standard deviation for the normal distribution in initialization.",
        dest="normal_stddev",
        default=1,
    )
    return parser


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    if options.train_file is not None and options.model_path is not None:
        model = unsupervised_morphology.UnsupervisedMorphology(
            input_file=options.train_file,
            smoothing_const=options.smooth_const,
            use_normal_init=options.normal_init,
            normal_mean=options.normal_mean,
            normal_stddev=options.normal_stddev,
        )
        print("Number of training words", len(model.params.word_counts))
        model.expectation_maximization(
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
                output = []
                for word in line.strip().split():
                    if word not in segment_cache:
                        segmented = segmentor.segment_word(
                            word, add_affix_symbols=options.add_affix_symbols
                        )
                        segment_cache[word] = segmented
                    output.append(segment_cache[word])
                writer.write(" ".join(output) + "\n")
        writer.close()
