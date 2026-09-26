"""Entry point so `python -m attention` prints the three commands."""


def main() -> None:
    print("Usage:")
    print("  python -m attention.train")
    print("  python -m attention.server")
    print('  python -m attention.translate "Hello."')


if __name__ == "__main__":
    main()
