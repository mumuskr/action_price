"""Minimal package health-check entry point."""

from brooks_trader import __version__


def main() -> None:
    """Print the installed package version and implemented phase."""
    print(f"brooks-trader {__version__} (Phase 10)")


if __name__ == "__main__":
    main()
