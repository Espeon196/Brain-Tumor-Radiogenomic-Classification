import argparse
import os
import yaml


parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('--run', help="Run number", type=int, required=True)
args = parser.parse_args()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(FILE_DIR, r"config\config_{:0=3}.yaml".format(args.run))) as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    print(args.run)
    print(config)