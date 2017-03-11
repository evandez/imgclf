package v2;

import static v2.Util.checkNotEmpty;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;
import static v2.Util.tensorAdd;

import java.util.ArrayList;
import java.util.List;

/**
 * A layer that performs n convolutions. Uses ReLU for activation.
 */
public class ConvolutionLayer implements PlateLayer {
    /**
     * Convolutions are laid out RGBG RGBG RGBG ... if numChannels = 4
     * or X X X ... if numChannels = 1
     */
    private final List<Plate> convolutions;
    private List<Plate> previousInput;
    private List<Plate> previousOutput;
    private int numChannels;

    private ConvolutionLayer(List<Plate> convolutions, int numChannels) {
        this.convolutions = convolutions;
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
        // Stores the delta values for all the plates in current layer
        List<Plate> deltaOutput = new ArrayList<>(errors.size());
        // Total error
        double[][][] error = new double[errors.size()][][];
        double[][][] delta = new double[error.length][][];

        for (int i = 0; i < errors.size(); i++) {
            error[i] = new double[previousInput.get(i).getHeight()][previousInput.get(i).getWidth()];
            /*// Stores the delta values for this plate
            double[][] deltaConvolutions = new double[errors.get(i).getHeight()][errors.get(i).getWidth()];
            // Stores the change in convolution values for plate i
            double[][] updateConvolutions = new double[errors.get(i).getHeight()][errors.get(i).getWidth()];

            // Loop over the entire plate and update weights (convolution values) using the equation
            // given in Russel and Norvig (check Lab 3 slides)
            for (int row = 0; row < errors.get(i).getHeight(); row++) {
                for (int col = 0; col < errors.get(i).getWidth(); col++) {
                    // TODO: Not sure if previousInput and previousOutput should be switched in the equations below
                    deltaConvolutions[row][col] = errors.get(i).valueAt(row, col)
                            * ActivationFunction.RELU.applyDerivative(previousOutput.get(i).valueAt(row, col))
                            * convolutions.get(i).valueAt(row, col);
                    updateConvolutions[row][col] = deltaConvolutions[row][col]
                            * previousInput.get(i).valueAt(row, col)
                            * learningRate;
                }
            }*/

            final int WINDOW_WIDTH = convolutions.get(i).getWidth();
            final int WINDOW_HEIGHT = convolutions.get(i).getHeight();
            for (int row = 0; row < error[i].length - WINDOW_HEIGHT; row++) {
                for (int col = 0; col < error[i][row].length - WINDOW_WIDTH; col++) {
                    for (int kernelRow = 0; kernelRow < WINDOW_HEIGHT; kernelRow++) {
                        for (int kernelCol = 0; kernelCol < WINDOW_WIDTH; kernelCol++) {
                            error[i][row + kernelRow][col + kernelCol] += errors.get(i).valueAt(row, col)
                                    * convolutions.get(i).valueAt(kernelRow, kernelCol);
                        }
                    }
                }
            }
            delta[i] = new double[error[i].length][error[i][0].length];
            for (int row = 0; row < error[i].length; row++) {
                for (int col = 0; col < error[i][row].length; col++) {
                    delta[i][row][col] += error[i][row][col]
                            * ActivationFunction.RELU.applyDerivative(previousInput.get(i).valueAt(row, col));
                }
            }
            deltaOutput.add(new Plate(delta[i]));
            /*// Update convolution values
            convolutions.get(i).setVals(
                    tensorAdd(convolutions.get(i).getValues(), updateConvolutions, false));
            // Add the delta values to list to be used by previous layer
            deltaOutput.add(new Plate(deltaConvolutions));*/
        }
        // TODO: I think this is what we should be returning but I'm not entirely sure
        return deltaOutput;
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
