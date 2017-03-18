package v2;

import java.util.function.Function;



/** Represents activation functions for any node. */
public enum ActivationFunction {
	RELU(   /* function */   x -> (x > 0) ? 1.0 * x : Math.pow(10, -7) * x,
			/* derivative */ x -> (x > 0) ? 1.0     : Math.pow(10, -7)),
	SIGMOID(/* function */   x -> 1.0 / (1.0 + Math.exp(-x)),
			/* derivative */ x -> (x * (1.0 - x)));

	private final Function<Double, Double> theFunc;
	private final Function<Double, Double> derivative;
	
    ActivationFunction(Function<Double, Double> theFunc, Function<Double, Double> derivative) {
    	this.theFunc = theFunc;
    	this.derivative = derivative;
    }

	/** Applies the activation function. */
    public double apply(double x) { return theFunc.apply(x); }
	
    /**
     * Evaluates the derivative at x. 
     * 
     * NOTE: Assumes that x is a value that has already been passed through the
     * activation function. (These derivatives all depend on the value at the activation function.)
     */
	public double applyDerivative(double x) { return derivative.apply(x); } 
}
