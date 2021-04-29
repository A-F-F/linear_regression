import argparse
import shelve


def predict(x: float, precision: int) -> None:
    sh = shelve.open("linear_regression")
    theta0, theta1 = sh.get("theta0"), sh.get("theta1")
    input_col, output_col = sh.get("input_col"), sh.get("output_col")
    sh.close()

    prediction = theta0 + theta1 * x
    print(f"Predicted {output_col} for {input_col} = {x} is {round(prediction, precision)}.")


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
        This program is meant make a prediction with a regression model of one variable, using
        parameters saved by its training program.
    """)
    parser.add_argument("x", type=float, help="Variable given to the prediction model.")
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        help="Precision of the returned prediction.",
        default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    predict(args.x, args.precision)
