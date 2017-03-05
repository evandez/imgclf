package v2;

import java.util.List;

public interface PlateLayer {
	int calculateNumOutputs(int numInputs);
	int calculateOutputHeight(int inputHeight);
	int calculateOutputWidth(int inputWidth);
	List<Plate> computeOutput(List<Plate> input);
	List<Plate> propagateError(List<Plate> errors, double learningRate);
}
