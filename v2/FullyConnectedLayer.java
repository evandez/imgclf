package v2;

import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;

/** 
 * Your standard fully-connected ANN.
 * 
 * This class stores the weights between inputs and nodes, and provides
 * functionality for computing the output given an input vector and for
 * backpropagating errors.
 */
public class FullyConnectedLayer {
	private final double[][] weights;
	private final ActivationFunction activation;
	
	private FullyConnectedLayer(double[][] weights, ActivationFunction activation) {
		this.weights = weights;
		this.activation = activation;
	}
	
	/** Compute the output of the given input vector. */
	public double[] computeOutput(double[] input) {
		if (input.length != weights[0].length) { // Valid check because we enforce > 0 inputs.
			throw new IllegalArgumentException(
					"Input length must match layer input specification.");
		}
		double[] outputs = new double[weights.length]; 
		for (int i = 0; i < outputs.length; i++) {
			double sum = 0;
			for (int j = 0; j < weights[i].length; j++) {
				sum += weights[i][j] * input[j];
			}
			// TODO: Add offset.
			outputs[i] = activation.apply(sum);
		}
		return outputs;
	}

	/** 
	 * Given the error from the previous layer, update the weights and return the error
	 * for this layer.
	 */
	public double[] propagateError(double[] error, double learningRate) {
		// TODO: Implement this method.
		return null;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("\n------\tFully Connected Layer\t------\n\n");
		builder.append(String.format("Number of inputs: %d\n", weights[0].length));
		builder.append(String.format("Number of nodes: %d\n", weights.length));
		builder.append(String.format("Activation function: %s\n", activation.toString()));
		builder.append("\n\t------------\t\n");
		return builder.toString();
	}
	
	/** Returns a new builder. */
	public static Builder newBuilder() { return new Builder(); }
	
	/** Simple builder pattern for organizing parameters. */
	public static class Builder {
		private ActivationFunction func = null;
		private int numInputs = 0;
		private int numNodes = 0;
		
		private Builder() {}

		public Builder setActivationFunction(ActivationFunction func) {
			checkNotNull(func, "Fully connected activation function");
			this.func = func;
			return this;
		}
		
		public Builder setNumInputs(int numInputs) {
			checkPositive(numInputs, "Number of fully connected inputs", false);
			this.numInputs = numInputs;
			return this;
		}
		
		public Builder setNumNodes(int numNodes) {
			checkPositive(numNodes, "Number of fully connected nodes", false);
			this.numNodes = numNodes;
			return this;
		}
		
		public FullyConnectedLayer build() {
			checkNotNull(func, "Fully connected activation function");
			checkPositive(numInputs, "Number of fully connected inputs", true);
			checkPositive(numNodes, "Number of fully connected nodes", true);
			double[][] weights = new double[numNodes][numInputs];
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					// TODO: Use Judy's weight initialization.
					weights[i][j] = Util.RNG.nextGaussian();
				}
			}
			return new FullyConnectedLayer(weights, func);
		}
	}
}
