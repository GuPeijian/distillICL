import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument(
        "--cl",
        action="store_true",
    )
    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    print(args.cl)
if __name__=="__main__":
    main()