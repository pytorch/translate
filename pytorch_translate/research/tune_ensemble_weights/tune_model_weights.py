#!/usr/bin/env python3

import itertools

import numpy as np
import pandas as pd
from fairseq import options
from pytorch_translate import generate
from pytorch_translate.constants import CHECKPOINT_PATHS_DELIMITER


def add_tune_args(parser):
    group = parser.add_argument_group("Tune parameter parser.")
    group.add_argument(
        "--n-grid",
        default=6,
        type=int,
        metavar="N",
        help="how many grid added to tune for each weight.",
    )
    group.add_argument(
        "--weight-lower-bound",
        default=0.0,
        type=float,
        help="lower bound for each weight.",
    )
    group.add_argument(
        "--weight-upper-bound",
        default=1.0,
        type=float,
        help="upper bound for each weight.",
    )
    group.add_argument(
        "--output-file-name",
        default="output.csv",
        type=str,
        help="name of output file.",
    )
    return parser


def tune_model_weights():
    parser = generate.get_parser_with_args()
    parser = add_tune_args(parser)
    args = options.parse_args_and_arch(parser)
    print(args.model_weights)
    n_models = len(args.path.split(CHECKPOINT_PATHS_DELIMITER))
    print(n_models)

    weight_grid = np.linspace(
        args.weight_lower_bound, args.weight_upper_bound, args.n_grid + 1
    )
    weight_vec_aux = list(itertools.product(weight_grid, weight_grid))
    weight_vec = []
    for w1, w2 in weight_vec_aux:
        weight_sum = w1 + w2
        if weight_sum <= 1:
            w3 = 1 - weight_sum
            weight_vec.append(str(w1) + "," + str(w2) + "," + str(w3))

    print(len(weight_vec))
    output = pd.DataFrame()
    for weight in weight_vec:
        args.model_weights = weight
        print(args.model_weights)
        generate.validate_args(args)
        score = generate.generate(args)
        print(score)
        output = output.append(
            {"weight": args.model_weights, "bleu_score": score}, ignore_index=True
        )
        output.to_csv(args.output_file_name)
    return output


if __name__ == "__main__":
    tune_model_weights()
