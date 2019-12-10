#!/usr/bin/env python3
import json

from ax.service.managed_loop import optimize
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
        "--num-trails-ax-opt",
        default=5,
        type=int,
        help="number of trials in AX optimization.",
    )
    group.add_argument(
        "--output-json-best-parameters",
        default="best_parameters.json",
        type=str,
        help="name of output file for the best parameters.",
    )
    group.add_argument(
        "--output-json-best-value",
        default="best_value.json",
        type=str,
        help="name of output file for the best value of the evaluation function.",
    )
    return parser


def tune_model_weights():
    parser = generate.get_parser_with_args()
    parser = add_tune_args(parser)
    args = options.parse_args_and_arch(parser)
    n_models = len(args.path.split(CHECKPOINT_PATHS_DELIMITER))
    print(n_models)
    print(args.weight_lower_bound)
    print(args.weight_upper_bound)
    print(args.output_json_best_parameters)
    print(args.output_json_best_value)
    print(args.num_trails_ax_opt)

    def evaluation_function(parameterization):
        w1 = parameterization.get("w1")
        w2 = parameterization.get("w2")
        w3 = parameterization.get("w3")
        weight = str(w1) + "," + str(w2) + "," + str(w3)
        args.model_weights = weight
        generate.validate_args(args)
        score = generate.generate(args)
        return {"bleu_score": (score, 0.0)}

    lower_bound = args.weight_lower_bound
    upper_bound = args.weight_upper_bound
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "w1",
                "type": "range",
                "bounds": [lower_bound, upper_bound],
                "value_type": "float",
            },
            {"name": "w2", "type": "range", "bounds": [lower_bound, upper_bound]},
            {"name": "w3", "type": "range", "bounds": [lower_bound, upper_bound]},
        ],
        experiment_name="tune_model_weights",
        objective_name="bleu_score",
        evaluation_function=evaluation_function,
        minimize=True,  # Optional, defaults to False.
        parameter_constraints=[
            "w1 + w2 + w3 <= 1",
            "w1 + w2 + w3 >= 0.99",
        ],  # Optional.
        total_trials=args.num_trails_ax_opt,  # Optional.
    )

    json_file = json.dumps(best_parameters)
    with open(args.output_json_best_parameters, "w") as f:
        f.write(json_file)
        f.close()

    json_file = json.dumps(values)
    with open(args.output_json_best_value, "w") as f:
        f.write(json_file)
        f.close()
    return best_parameters, values


if __name__ == "__main__":
    tune_model_weights()
