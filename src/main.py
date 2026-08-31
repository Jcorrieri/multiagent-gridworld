import argparse

from test import main as test_main
from train import main as train_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test"], required=True)
    parser.add_argument('--config', type=str, default='default')
    args = parser.parse_args()

    if args.mode == "train":
        train_main(args)
    else:
        test_main(args)


if __name__ == "__main__":
    main()
