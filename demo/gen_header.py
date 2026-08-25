from profile_utils import ExperimentOutput, AttentionOutput
import argparse

OUTPUTS = {
    "ExperimentOutput": ExperimentOutput,
    "AttentionOutput": AttentionOutput,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        choices=list(OUTPUTS),
    )
    args = parser.parse_args()
    out = OUTPUTS[args.method]
    print(out.list_to_csv(out.header()))
