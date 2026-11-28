"""Run the confirmed nodal short-axis 1.25 protocol on selected lesion IDs only."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     usage="%(prog)s {tiny,medsam2} [options]")
    parser.add_argument("model", choices=("tiny", "medsam2"))
    argv = sys.argv[1:] if argv is None else argv
    model = parser.parse_args(argv[:1]).model
    if model == "tiny":
        from infer_effmedsam2 import main as run
    else:
        from infer_medsam2 import main as run
    return run(argv[1:], nodal=True)


if __name__ == "__main__":
    sys.exit(main())
