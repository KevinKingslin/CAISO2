"""Verify every packaged CAISO prompt without making model calls."""

import argparse

from caiso.prompts import PromptResourceError, validate_resources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        count = validate_resources()
    except PromptResourceError as error:
        parser.exit(1, str(error) + "\n")
    print(f"Verified {count} CAISO prompt files")


if __name__ == "__main__":
    main()
