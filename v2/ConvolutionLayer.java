package v2;

import static v2.Util.checkNotEmpty;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;

import java.util.ArrayList;
import java.util.List;

/** A layer that performs n convolutions. Uses ReLU for activation. */
public class ConvolutionLayer implements PlateLayer {
	private final List<Plate> convolutions;
	private final List<Plate> previousInput;
	private final List<Plate> previousOutput;
	
	private ConvolutionLayer(List<Plate> convolutions) {
		this.convolutions = convolutions;
		this.previousInput = new ArrayList<>();
		this.previousOutput = new ArrayList<>();
	}

	@Override
	public int calculateNumOutputs(int numInputs) {
		return numInputs * convolutions.size();
	}
	
	@Override
	public int calculateOutputHeight(int inputHeight) {
		return inputHeight;
	}
	
	@Override
	public int calculateOutputWidth(int inputWidth) {
		return inputWidth;
	}

	@Override
	public List<Plate> computeOutput(List<Plate> input) {
		checkNotNull(input, "Convolution layer input");
		checkNotEmpty(input, "Convolution layer input", false);
		
		// Convolve each input with each mask.
		List<Plate> output = new ArrayList<>();
		for (Plate mask : convolutions) {
			for (Plate inputPlate : input) {
				output.add(inputPlate.convolve(mask));
			}
		}
		
		// Activate each output.
		for (Plate inputPlate : input) {
			output.add(inputPlate.applyActivation(ActivationFunction.RELU));
		}
		
		return output;
	}
	
	@Override
	public List<Plate> propagateError(List<Plate> errors, double learningRate) {
		if (errors.size() != previousOutput.size() || previousInput.isEmpty()) {
			throw new IllegalArgumentException("Bad propagation state.");
		}
		
		// TODO: Implement this method.
		return null;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("\n------\tConvolution Layer\t------\n\n");
		builder.append(String.format(
				"Convolution Size: %dx%dx%d\n",
				convolutions.get(0).getNumChannels(),
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
			checkPositive(numChannels, "Number of channels", false);
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
			checkPositive(convolutionHeight, "Convolution height", true);
			checkPositive(convolutionWidth, "Convolution width", true);
			checkPositive(numConvolutions, "Number of convolutions", true);
			List<Plate> convolutions = new ArrayList<>();
			for (int i = 0; i < numConvolutions; i++) {
				convolutions.add(
						new Plate(
								createRandomConvolution(numChannels, convolutionHeight, convolutionWidth)));
			}
			return new ConvolutionLayer(convolutions);
		}
		
		// TODO: We should probably use the initialization method suggested by Judy.
		private static double[][][] createRandomConvolution(int numChannels, int height, int width) {
			double[][][] plateValues = new double[numChannels][height][width];
			for (int i = 0; i < plateValues.length; i++) {
				for (int j = 0; j < plateValues[i].length; j++) {
					for (int k = 0; k < plateValues[i][j].length; k++) {
						plateValues[i][j][k] = Util.RNG.nextGaussian();
					}
				}
			}
			return plateValues;
		}
	}
}
