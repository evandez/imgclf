package v2;

import java.util.ArrayList;
import java.util.List;

import static v2.Util.*;

/** A layer that performs n convolutions. Uses ReLU for activation. */
public class ConvolutionLayer implements PlateLayer {
	/** 
	 * Convolutions are laid out RGBG RGBG RGBG ... if numChannels = 4
	 * or X X X ... if numChannels = 1
	 */
	private final List<Plate> convolutions;
	private final List<Plate> previousInput;
	private final List<Plate> previousOutput;
	private int numChannels;
	
	private ConvolutionLayer(List<Plate> convolutions, int numChannels) {
		this.convolutions = convolutions;
		this.numChannels = numChannels;
		this.previousInput = new ArrayList<>();
		this.previousOutput = new ArrayList<>();
	}

	public int numConvolutions() {
		return convolutions.size();
	}
	
	@Override
	public int calculateNumOutputs(int numInputs) {
		return numInputs / numChannels;
	}
	
	@Override
	public int calculateOutputHeight(int inputHeight) {
		return inputHeight - convolutions.get(0).getHeight() + 1;
	}
	
	@Override
	public int calculateOutputWidth(int inputWidth) {
		return inputWidth - convolutions.get(0).getWidth() + 1;
	}

	@Override
	public List<Plate> computeOutput(List<Plate> input) {
		checkNotNull(input, "Convolution layer input");
		checkNotEmpty(input, "Convolution layer input", false);
		
		// Convolve each input with each mask.
		List<Plate> output = new ArrayList<>();
		Plate[] masks = new Plate[numChannels];
		int maskHeight = convolutions.get(0).getHeight();
		int maskWidth = convolutions.get(0).getWidth();
		for (int i = 0; i < convolutions.size(); i += numChannels) {
			double[][] values = new double[input.get(0).getHeight() - maskHeight + 1][input.get(0).getWidth() - maskWidth + 1];
			// convolve each input image, sum the output, add the new plate
			for (int j = 0; j < numChannels; j++) {
				masks[j] = convolutions.get(i + j);
				Util.tensorAdd(values, input.get(j).convolve(masks[j]).getVals(), true);
				input.get(j);
			}
			output.add((new Plate(values).applyActivation(ActivationFunction.RELU)));
		}
		return output;
	}
	
	@Override
	public List<Plate> propagateError(List<Plate> errors, double learningRate) {
		if (errors.size() != previousOutput.size() || previousInput.isEmpty()) {
			throw new IllegalArgumentException("Bad propagation state.");
		}

        for (int i = 0; i < errors.size(); i++) {

		    // Holds the change in convolution values for plate i
		    double[][] deltaConvolutions = new double[errors.get(i).getHeight()][errors.get(i).getWidth()];

		    // Loop over the entire plate and update weights (convolution values) using the equation
            // given in Russel and Norvig (check Lab 3 slides)
            for (int row = 0; row < errors.get(i).getHeight(); row++)
                for (int col = 0; col < errors.get(i).getWidth(); col++)
                    deltaConvolutions[row][col] = errors.get(i).valueAt(row, col)
                            * ActivationFunction.RELU.applyDerivative(previousInput.get(i).valueAt(row, col))
                            * previousOutput.get(i).valueAt(row, col)
                            * convolutions.get(i).valueAt(row, col)
                            * learningRate;

            // Update convolution values
            convolutions.get(i).setVals(
                    tensorAdd(convolutions.get(i).getVals(), deltaConvolutions, false));
        }
        return convolutions;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("\n------\tConvolution Layer\t------\n\n");
		builder.append(String.format(
				"Convolution Size: %dx%dx%d\n",
				numChannels,
				convolutions.get(0).getHeight(),
				convolutions.get(0).getWidth()));
		builder.append(String.format("Number of convolutions: %d\n", convolutions.size()));
		builder.append("Activation Function: RELU\n");
		builder.append("\n\t------------\t\n");
		return builder.toString();
	}
	
	/** Returns a new builder. */
	public static Builder newBuilder() { return new Builder(); }
	
	/** A simple builder pattern for managing the layer's parameters at construction. */
	public static class Builder {
		private int numChannels = 0;
		private int convolutionHeight = 0;
		private int convolutionWidth = 0;
		private int numConvolutions = 0;
		
		private Builder() {}

		public Builder setConvolutionSize(int numChannels, int height, int width) {
			checkPositive(numChannels, "Convolution channels", false);
			checkPositive(height, "Convolution height", false);
			checkPositive(width, "Convolution width", false);
			this.numChannels = numChannels;
			this.convolutionHeight = height;
			this.convolutionWidth = width;
			return this;
		}
		
		public Builder setNumConvolutions(int numConvolutions) {
			checkPositive(numConvolutions, "Number of convolutions", false);
			this.numConvolutions = numConvolutions;
			return this;
		}
		
		public ConvolutionLayer build() {
			checkPositive(numChannels, "Convolution channels", true);
			checkPositive(convolutionHeight, "Convolution height", true);
			checkPositive(convolutionWidth, "Convolution width", true);
			checkPositive(numConvolutions, "Number of convolutions", true);
			List<Plate> convolutions = new ArrayList<>();
			for (int i = 0; i < numConvolutions; i++) {
				for (int j = 0; j < numChannels; j++) {
					convolutions.add(
							new Plate(
									createRandomConvolution(convolutionHeight, convolutionWidth)));
				}
			}
			return new ConvolutionLayer(convolutions, numChannels);
		}
		
		// TODO: We should probably use the initialization method suggested by Judy.
		private static double[][] createRandomConvolution(int height, int width) {
			double[][] plateValues = new double[height][width];
			for (int i = 0; i < plateValues.length; i++) {
				for (int j = 0; j < plateValues[i].length; j++) {
					plateValues[i][j] = Util.RNG.nextGaussian();
				}
			}
			return plateValues;
		}
	}
}
