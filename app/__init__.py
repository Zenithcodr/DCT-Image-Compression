"""Convenience launcher: `python -m app.gui` or `python app/gui.py`.

We import the GUI lazily so the rest of the project can be tested on
systems where Tk is not installed (common on minimal Python builds)."""

import sys

try:
    from .gui import DCTApp, main  # noqa: F401
except ImportError:
    DCTApp = None  # type: ignore[assignment]

    def main() -> int:  # pragma: no cover
        print(
            "Tkinter is not installed in this Python distribution.\n"
            "On Windows try reinstalling Python from python.org with the default\n"
            "options, which includes tkinter. On Linux: 'sudo apt install python3-tk'."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
