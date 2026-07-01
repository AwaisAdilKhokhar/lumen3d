import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # the "reconstruct" window
    recon = subparsers.add_parser("reconstruct")
    recon.add_argument("input")              # the input video/folder (positional)
    recon.add_argument("-o", "--output", default="scene")   # where to save

    args = parser.parse_args()

    if args.command == "reconstruct":
        print(f"would reconstruct {args.input} -> {args.output}")


if __name__ == "__main__":
    main()