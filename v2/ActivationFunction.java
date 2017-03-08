package v2;

import java.util.function.Function;

/** Represents activation functions for any node. */
public enum ActivationFunction {
	RELU(x -> Math.max(x, .01 * x)),
	RELUPRIME(x -> (x >= 0) ? 1 : .01),
	SIGMOID(x -> 1 / (1 + Math.pow(Math.E, -x))),
	SIGMOIDPRIME(x -> (x) * (1-x));

	private final Function<Double, Double> func;

    ActivationFunction(Function<Double, Double> func) { this.func = func; }

	public double apply(double x) { return func.apply(x); }
}
