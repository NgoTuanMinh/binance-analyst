"""Allow ``python -m features.pipeline`` via ``python -m features`` (delegates to pipeline CLI)."""

from features.pipeline import main

if __name__ == "__main__":
    main()
