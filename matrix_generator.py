import argparse
from random import randint


def args_parser():
    parser = argparse.ArgumentParser(description="Matrix generator to a .txt file")
    parser.add_argument("-r", "--rows", type=int, required=True, help="Number of rows")
    parser.add_argument("-c", "--cols", type=int, required=True, help="Number of columns")
    parser.add_argument("-m", "--min", type=int, default=0, help="Minimum value")
    parser.add_argument("-M", "--max", type=int, default=100, help="Maximum value")
    parser.add_argument("-o", "--output", default="matrix.txt", help="Output file name")
    return parser


def generate_matrix(rows, cols, min_val, max_val, filename):
    if rows <= 0 or cols <= 0:
        raise ValueError("Matrix dimensions must be greater than 0")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{rows}\t{cols}\n")
        for _ in range(rows):
            row = [str(randint(min_val, max_val)) for _ in range(cols)]
            f.write(" ".join(row) + "\n")


def main():
    args = args_parser().parse_args()

    if args.min > args.max:
        raise ValueError("Minimum value cannot be greater than maximum value")

    generate_matrix(args.rows, args.cols, args.min, args.max, args.output)


if __name__ == "__main__":
    main()