package v2;

import java.util.function.Function;

public enum ActivationFunction {
	RELU(x -> { return Math.max(x, 0); }),
	SIGMOID(x -> 1 / (1 + Math.pow(Math.E, -x)));

	private final Function<Double, Double> func;

	ActivationFunction(Function<Double, Double> func) { this.func = func; }

	public double apply(double x) { return func.apply(x); }
}
