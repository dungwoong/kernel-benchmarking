from profile_utils import ExperimentOutput
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", 
        choices=["ExperimentOutput"],
    )
    args = parser.parse_args()
    if args.method == "ExperimentOutput":
        print(ExperimentOutput.list_to_csv(ExperimentOutput.header()))
