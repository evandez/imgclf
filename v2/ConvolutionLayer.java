package v2;

import static v2.Util.checkNotEmpty;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;
import static v2.Util.doubleArrayCopy2D;
import static v2.Util.tensorAdd;

import java.util.ArrayList;
import java.util.List;

/**
 * A layer that performs n convolutions. Uses ReLU for activation.
 */
public class ConvolutionLayer implements PlateLayer {
    // Convolutions are laid out RGBG RGBG RGBG ... if numChannels = 4
    // or X X X ... if numChannels = 1
    private final List<Plate> convolutions;
    private final List<Plate> savedConvolutions;
    private List<Plate> previousInput;
    private List<Plate> previousOutput;
    private int numChannels;

    private ConvolutionLayer(List<Plate> convolutions, int numChannels) {
        this.convolutions = convolutions;
        this.savedConvolutions = new ArrayList<>(deepCopyPlates(convolutions));
        this.numChannels = numChannels;
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

    public List<Plate> getConvolutions() {
        return convolutions;
    }

    @Override
    public List<Plate> computeOutput(List<Plate> input) {
        checkNotNull(input, "Convolution layer input");
        checkNotEmpty(input, "Convolution layer input", false);
        previousInput = input;
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
                Util.tensorAdd(values, input.get(j).convolve(masks[j]).getValues(), true);
                input.get(j);
            }
            output.add((new Plate(values).applyActivation(ActivationFunction.RELU)));
        }
        previousOutput = output;
        return output;
    }

    @Override
    public List<Plate> propagateError(List<Plate> errors, double learningRate) {
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
//                System.out.println(deltaOutput.get(deltaOutput.size() - 1).toString());
            }
            return deltaOutput;
        } else {
            return null;
        }
    }

    @Override
    public void saveState() { 
    	savedConvolutions.clear();
    	savedConvolutions.addAll(deepCopyPlates(convolutions));
    }

    @Override
    public void restoreState() {
    	convolutions.clear();
    	convolutions.addAll(savedConvolutions);
    }

    private static List<Plate> deepCopyPlates(List<Plate> plates) {
    	List<Plate> deepCopy = new ArrayList<>(plates.size());
    	for (Plate plate : plates) {
    		double[][] plateValues = plate.getValues();
    		double[][] copiedValues = new double[plateValues.length][plateValues[0].length];
    		doubleArrayCopy2D(plateValues, copiedValues);
    		deepCopy.add(new Plate(copiedValues));
    	}
    	return deepCopy;
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
