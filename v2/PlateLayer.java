package v2;

import java.util.List;

/** Interface for passing plates between conv and pool layers. */
public interface PlateLayer {

	/** Returns the number of plates in this layer. */
	int getSize();
	
	/** Given the number of inputs, return how many plates this layer will output. */
	int calculateNumOutputs(int numInputs);
	
	int calculateNumOutputs();
	
	/** Given the height of the input, return the height of the output. */
	int calculateOutputHeight(int inputHeight);
	
	int calculateOutputHeight();
	
	/** Given the width of the input, return the width of the output. */
	int calculateOutputWidth(int inputWidth);
	
	int calculateOutputWidth();
	
	/** Pass the given plates through this layer. */
	List<Plate> computeOutput(List<Plate> input, boolean currentlyTraining);

	/** 
	 * Propagate errors (deltas stored in plates) through this layer,
	 * and return the deltas for the next layer.
	 */
	List<Plate> propagateError(List<Plate> errors, double learningRate);
	
	/** Saves the current state of the layer (weights, etc.) */
	void saveState();

	/** Restores the last saved state of the layer. */
	void restoreState();
}
