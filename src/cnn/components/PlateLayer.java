package cnn.components;

import java.util.List;

/** Interface for passing plates between conv and pool layers. */
public interface PlateLayer {
	/** Given the number of inputs, return how many plates this layer will output. */
	int calculateNumOutputs(int numInputs);
	
	/** Given the height of the input, return the height of the output. */
	int calculateOutputHeight(int inputHeight);
	
	/** Given the width of the input, return the width of the output. */
	int calculateOutputWidth(int inputWidth);
	
	/** Pass the given plates through this layer. */
	List<Plate> computeOutput(List<Plate> input);
	
	/** 
	 * Propagate errors (deltas stored in plates) through this layer,
	 * and return the deltas for the next layer.
	 */
	List<Plate> propagateError(List<Plate> errors, double learningRate);
}
