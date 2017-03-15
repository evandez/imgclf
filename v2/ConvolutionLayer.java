package v2;

<<<<<<< HEAD
import static v2.Util.checkNotEmpty;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;

=======
>>>>>>> parent of 63e74f8... Some fixes and changes, prop for Pooling and Conv both wrong
import java.util.ArrayList;
import java.util.List;

import static v2.Util.*;

<<<<<<< HEAD
/**
 * A layer that performs n convolutions. Uses ReLU for activation.
 */
public class ConvolutionLayer implements PlateLayer {
    /**
     * Convolutions are laid out RGBG RGBG RGBG ... if numChannels = 4
     * or X X X ... if numChannels = 1
     */
    private final List<List<Plate>> convolutions;
    private List<Plate> previousInput;
    private List<Plate> previousOutput;

    private ConvolutionLayer(List<List<Plate>> convolutions) {
        this.convolutions = convolutions;
    }

    public int numConvolutions() {
        return convolutions.size();
    }

    /** Returns the number of channels in each convolution. */
    public int getConvolutionDepth() {
    	return convolutions.get(0).size();
    }
    
    public int getConvolutionHeight() {
    	return convolutions.get(0).get(0).getHeight();
    }

    public int getConvolutionWidth() {
    	return convolutions.get(0).get(0).getWidth();
    }
    
    @Override
    public int calculateNumOutputs(int numInputs) {
        return numInputs / getConvolutionDepth();
    }

    @Override
    public int calculateOutputHeight(int inputHeight) {
        return inputHeight - getConvolutionHeight() + 1;
    }

    @Override
    public int calculateOutputWidth(int inputWidth) {
        return inputWidth - getConvolutionWidth() + 1;
    }

    public List<List<Plate>> getConvolutions() {
        return convolutions;
    }

    @Override
    public List<Plate> computeOutput(List<Plate> input) {
        checkNotNull(input, "Convolution layer input");
        checkNotEmpty(input, "Convolution layer input", false);
        previousInput = input;
        // Convolve each input with each mask.
        List<Plate> output = new ArrayList<>();
        for (int i = 0; i < convolutions.size(); i ++) {
            double[][] values = new double[input.get(0).getHeight() - getConvolutionHeight() + 1][input.get(0).getWidth() - getConvolutionWidth() + 1];
            // convolve each input image, sum the output, add the new plate
            for (int j = 0; j < getConvolutionDepth(); j++) {
                Plate convolution = convolutions.get(i).get(j);
                Util.tensorAdd(values, input.get(j).convolve(convolution).getValues(), true);
                input.get(j);
            }
            output.add((new Plate(values).applyActivation(ActivationFunction.RELU)));
        }
        previousOutput = output;
        return output;
    }

    @Override
    public List<Plate> propagateError(List<Plate> errors, double learningRate) {
//   	 	System.out.println(this);
        if (errors.size() != previousOutput.size() || previousInput.isEmpty()) {
            throw new IllegalArgumentException("Bad propagation state.");
        }

        // Update the convolution values
        for (int i = 0; i < previousInput.size(); i++) {
            // Loop over the plate
            for (int j = 0; j <= errors.get(i).getHeight() - convolutions.get(i).getHeight(); j++) {
                for (int k = 0; k <= errors.get(i).getWidth() - convolutions.get(i).getWidth(); k++) {
                    double[][] update = new double[convolutions.get(i).getHeight()][convolutions.get(i).getWidth()];
                    for (int l = 0; l < convolutions.get(i).getHeight(); l++) {
                        for (int m = 0; l < convolutions.get(i).getHeight(); l++) {
                            if (update[l][m] == 0)
                                update[l][m] = 1;
                            update[l][m] = previousInput.get(i).valueAt(j+l, k+m)
                                    * errors.get(i).valueAt(j+l, k+m)
                                    * learningRate;
                        }
                    }
                    convolutions.get(i).setVals(tensorAdd(convolutions.get(i).getValues(), update, false));
                }
            }
        }

        // Stores the delta values for all the plates in current layer
        List<Plate> deltaOutput = new ArrayList<>(errors.size());
        // Total error
        double[][][] error = new double[errors.size()][][];
        double[][][] delta = new double[error.length][][];

        if (previousInput.size() == errors.size()) {
            for (int i = 0; i < errors.size(); i++) {
                error[i] = new double[previousInput.get(i).getHeight()][previousInput.get(i).getWidth()];

//                // Stores the delta values for this plate
//                double[][] deltaConvolutions = new double[errors.get(i).getHeight()][errors.get(i).getWidth()];
//                // Stores the change in convolution values for plate i
//                double[][] updateConvolutions = new double[errors.get(i).getHeight()][errors.get(i).getWidth()];
                // Loop over the entire plate and update weights (convolution values) using the equation
                // given in Russel and Norvig (check Lab 3 slides)
                delta[i] = new double[previousInput.get(i).getHeight()][previousInput.get(i).getWidth()];

                for (int row = 0; row <= delta[i].length - convolutions.get(i).getHeight(); row++) {
                    for (int col = 0; col <= delta[i][row].length - convolutions.get(i).getWidth(); col++) {
                        for (int kernelRow = 0; kernelRow < convolutions.get(i).getHeight(); kernelRow++) {
                            for (int kernelCol = 0; kernelCol < convolutions.get(i).getWidth(); kernelCol++) {
                                delta[i][row + kernelRow][col + kernelCol] += errors.get(i).valueAt(row, col)
                                        * convolutions.get(i).rot180().valueAt(kernelRow, kernelCol);
                            }
                        }
                    }
                }

                for (int row = 0; row < delta[i].length; row++) {
                    for (int col = 0; col < delta[i][row].length; col++) {
                        delta[i][row][col] *= ActivationFunction.RELU.applyDerivative(previousInput.get(i).valueAt(row, col));
                    }
                }

                deltaOutput.add(new Plate(delta[i]));
                // Update convolution values
                //convolutions.get(i).setVals(
                 //       tensorAdd(convolutions.get(i).getValues(), updateConvolutions, false));
                // Add the delta values to list to be used by previous layer
                //deltaOutput.add(new Plate(deltaConvolutions));
            }
            return deltaOutput;
        } else {
            return null;
//
//        for (int i = 0; i < errors.size(); i++) {
//            error[i] = new double[previousInput.get(0).getHeight()][previousInput.get(0).getWidth()];
//            delta[i] = new double[error[i].length][error[i][0].length];
//            for (int kernel = 0; kernel < getConvolutionDepth(); kernel++) {
//	            for (int row = 0; row < error[i].length - getConvolutionHeight(); row++) {
//	                for (int col = 0; col < error[i][row].length - getConvolutionWidth(); col++) {
//	                    for (int kernelRow = 0; kernelRow < getConvolutionHeight(); kernelRow++) {
//	                        for (int kernelCol = 0; kernelCol < getConvolutionWidth(); kernelCol++) {
//	                            error[i][row + kernelRow][col + kernelCol] += errors.get(i).valueAt(row, col)
//	                                    * convolutions.get(i).get(kernel).valueAt(kernelRow, kernelCol);
//	                        }
//	                    }
//	                }
//	            }
//	            for (int row = 0; row < error[i].length; row++) {
//	                for (int col = 0; col < error[i][row].length; col++) {
//	                    delta[i][row][col] += error[i][row][col]
//	                            * ActivationFunction.RELU.applyDerivative(previousInput.get(kernel).valueAt(row, col));
//	                }
//	            }
//            }
//            deltaOutput.add(new Plate(delta[i]));
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("\n------\tConvolution Layer\t------\n\n");
        builder.append(String.format(
                "Convolution Size: %dx%dx%d\n",
                getConvolutionDepth(),
                getConvolutionHeight(),
                getConvolutionWidth()));
        builder.append(String.format("Number of convolutions: %d\n", convolutions.size()));
        builder.append("Activation Function: RELU\n");
        builder.append("\n\t------------\t\n");
        return builder.toString();
    }

    /**
     * Returns a new builder.
     */
    public static Builder newBuilder() {
        return new Builder();
    }

    /**
     * A simple builder pattern for managing the layer's parameters at construction.
     */
    public static class Builder {
        private int numChannels = 0;
        private int convolutionHeight = 0;
        private int convolutionWidth = 0;
        private int numConvolutions = 0;

        private Builder() {
        }

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
            List<List<Plate>> convolutions = new ArrayList<>();
            for (int i = 0; i < numConvolutions; i++) {
            	List<Plate> channelConvolutions = new ArrayList<>();
                for (int j = 0; j < numChannels; j++) {
                    channelConvolutions.add(
                            new Plate(
                                    createRandomConvolution(convolutionHeight, convolutionWidth)));
                }
                convolutions.add(channelConvolutions);
            }
            return new ConvolutionLayer(convolutions);
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
=======
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
>>>>>>> parent of 63e74f8... Some fixes and changes, prop for Pooling and Conv both wrong
}
