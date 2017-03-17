package v2;

import static v2.Util.RNG;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;
import static v2.Util.doubleArrayCopy2D;
import static v2.Util.outerProduct;
import static v2.Util.scalarMultiply;
import static v2.Util.tensorSubtract;

/** 
 * Your standard fully-connected ANN.
 * 
 * This class stores the weights between inputs and nodes, and provides
 * functionality for computing the output given an input vector and for
 * back-propagating errors.
 */
public class FullyConnectedLayer {
	private final double[][] weights;
	private final double[][] savedWeights;
	private final double[] lastInput;
	private final double[] lastOutput;
	private final ActivationFunction activation;
	private final double dropoutRate;
	private final boolean[] activeNodes;

	private FullyConnectedLayer(
			double[][] weights,
			ActivationFunction activation,
			double dropoutRate) {
		// Initialize the standard stuff.
		this.weights = weights;
		this.savedWeights = new double[weights.length][weights[0].length];
		this.lastInput = new double[weights[0].length];
		this.lastOutput = new double[weights.length];
		this.activation = activation;
		
		// Initialize auxiliary data for dropout.
		this.dropoutRate = dropoutRate;
		this.activeNodes = new boolean[weights.length];
		resetDroppedOutNodes(); // Nodes are active by default.
		
		// Set the last value to be the offset. This will never change.
		this.lastInput[this.lastInput.length - 1] = -1;

		// Save initial weights.
		doubleArrayCopy2D(weights, savedWeights);
	}

	/** Compute the output of the given input vector. */
	public double[] computeOutput(double[] input, boolean currentlyTraining) {
		if (input.length != lastInput.length - 	1) {
			throw new IllegalArgumentException(
					String.format(
							"Input length in fully connected layer was %d, should be %d.",
							input.length,
							lastInput.length));
		}
		
		if (currentlyTraining) {
			determineDroppedOutNodes();
		} else {
			resetDroppedOutNodes();
		}
		
		System.arraycopy(input, 0, lastInput, 0, input.length);
		for (int i = 0; i < lastOutput.length; i++) {
			// Skip nodes that are dropped out.
			if (!activeNodes[i]) {
				continue;
			}

			double sum = 0;
			for (int j = 0; j < lastInput.length; j++) {
				double sumTerm = weights[i][j] * lastInput[j];
				if (!currentlyTraining) {
					// Down-scale only if testing/tuning.
					sumTerm *= (1 - dropoutRate);
				}
				sum += sumTerm;
			}
			lastOutput[i] = activation.apply(sum);
		}
		return lastOutput;
	}

    /**
     * Given the error from the previous layer, update the weights and return the error
     * for this layer.
     */
    double[] propagateError(double[] proppedDelta, double learningRate) {
        if (proppedDelta.length != weights.length) {
            throw new IllegalArgumentException(
                    String.format(
                            "Got length %d delta, expected length %d!",
                            proppedDelta.length,
                            weights.length));
        }

        // Compute deltas for the next layer.
        double[] delta = new double[weights[0].length - 1]; // Don't count the offset here.
        for (int j = 0; j < delta.length; j++) {
            for (int i = 0; i < weights.length; i++) {
            	if (!activeNodes[i]) {
            		continue;
            	}
                delta[j] += proppedDelta[i] * weights[i][j] * activation.applyDerivative(lastInput[j]);
            }
        }

        // Update the weights using the propped delta.
        tensorSubtract(
                weights,
                scalarMultiply(
                        learningRate,
                        outerProduct(proppedDelta, lastInput),
                        true /* inline */),
                true /* inline */);
        return delta;
    }
    
    private void determineDroppedOutNodes() {
    	resetDroppedOutNodes();
    	for (int i = 0; i < activeNodes.length; i++) {
    		if (dropoutRate > RNG.nextDouble()) {
    			activeNodes[i] = false;
    		}
    	}
    }
    
    private void resetDroppedOutNodes() {
    	for (int i = 0; i < activeNodes.length; i++) {
    		activeNodes[i] = true;
    	}
    }
    
    /** Saves the current weights in an auxiliary array. */
    void saveWeights() { doubleArrayCopy2D(weights, savedWeights); }
    
    /** Restores the weights from the last save. */
    void restoreWeights() { doubleArrayCopy2D(savedWeights, weights); }
	
	@Override
	public String toString() {
        return "\n------\tFully Connected Layer\t------\n\n" +
                String.format("Number of inputs: %d (plus a bias)\n", weights[0].length - 1) +
                String.format("Number of nodes: %d\n", weights.length) +
                String.format("Activation function: %s\n", activation.toString()) +
                String.format("Dropout rate: %.2f", dropoutRate) +
                "\n\t------------\t\n";
	}
	
	/** Returns a new builder. */
	static Builder newBuilder() { return new Builder(); }
	
	/** Simple builder pattern for organizing parameters. */
	static class Builder {
		private ActivationFunction func = null;
		private int numInputs = 0;
		private int numNodes = 0;
		private double dropoutRate = 0;
		
		private Builder() {}

		Builder setActivationFunction(ActivationFunction func) {
			checkNotNull(func, "Fully connected activation function");
			this.func = func;
			return this;
		}
		
		Builder setNumInputs(int numInputs) {
			checkPositive(numInputs, "Number of fully connected inputs", false);
			this.numInputs = numInputs;
			return this;
		}
		
		Builder setNumNodes(int numNodes) {
			checkPositive(numNodes, "Number of fully connected nodes", false);
			this.numNodes = numNodes;
			return this;
		}
		
		Builder setDropoutRate(double dropoutRate) {
			if (dropoutRate < 0 || dropoutRate > 1) {
        		throw new IllegalArgumentException(
        				String.format("Invalid dropout rate of %.2f\n", dropoutRate));
        	}
			this.dropoutRate = dropoutRate;
			return this;
		}
		
		FullyConnectedLayer build() {
			checkNotNull(func, "Fully connected activation function");
			checkPositive(numInputs, "Number of fully connected inputs", true);
			checkPositive(numNodes, "Number of fully connected nodes", true);
			// Dropout rate defaults to 0.

			double[][] weights = new double[numNodes][numInputs + 1];
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Lab3.getRandomWeight(numInputs, numNodes);
				}
			}
			return new FullyConnectedLayer(weights, func, dropoutRate);
		}
	}
}
