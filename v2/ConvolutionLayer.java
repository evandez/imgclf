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
    private int numOutputs;
    private int outputHeight;
    private int outputWidth;
    private final double dropoutRate;
    private final List<boolean[][]> activeNodes;
    private double[][] update;
    private List<Plate> deltaOutput;
    private List<Plate> output;

    private ConvolutionLayer(List<Plate> convolutions, int numChannels, double dropoutRate) {
        this.convolutions = convolutions;
        this.savedConvolutions = new ArrayList<>(deepCopyPlates(convolutions));
        this.numChannels = numChannels;
        
        // Initialize dropout fields.
        this.dropoutRate = dropoutRate;
        this.activeNodes = new ArrayList<>(convolutions.size());
        for (Plate convolution : this.convolutions) {
        	this.activeNodes.add(new boolean[convolution.getHeight()][convolution.getWidth()]);
        }
        resetDroppedOutNodes();
    }

    @Override
    public int getSize() {
        return convolutions.size();
    }

    @Override
    public int calculateNumOutputs(int numInputs) {
    	numOutputs = numInputs / numChannels;
        return numOutputs;
    }
    
    @Override
    public int calculateNumOutputs() {
    	if (numOutputs > 0) {
    		return numOutputs;
    	} else {
    		throw new RuntimeException(
    				"Cannot call calculateNumOutputs without arguments before calling it with arguments.");
    	}
    }

    @Override
    public int calculateOutputHeight(int inputHeight) {
    	outputHeight = inputHeight - convolutions.get(0).getHeight() + 1;
        return outputHeight;
    }

    @Override
    public int calculateOutputHeight() {
    	if (outputHeight > 0) {
    		return outputHeight;
    	} else {
    		throw new RuntimeException(
    				"Cannot call calculateOutputHeight without arguments before calling it with arguments.");
    	}
    }
    
    @Override
    public int calculateOutputWidth(int inputWidth) {
    	outputWidth = inputWidth - convolutions.get(0).getWidth() + 1;
        return outputWidth;
    }
    
    @Override
    public int calculateOutputWidth() {
    	if (outputWidth > 0) {
    		return outputWidth;
    	} else {
    		throw new RuntimeException("Cannot call calculateOutputWidth without arguments before calling it with arguments.");
    	}
    }
    
    @Override
    public List<Plate> computeOutput(List<Plate> input, boolean currentlyTraining) {
    	if (!Lab3.RUNNING) {
    		checkNotNull(input, "Convolution layer input");
    		checkNotEmpty(input, "Convolution layer input", false);
    	}

        previousInput = deepCopyPlates(input);

		if (currentlyTraining) {
			determineDroppedOutNodes();
		} else {
			resetDroppedOutNodes();
		}

        if (output == null) {
        	int maskHeight = convolutions.get(0).getHeight();
        	int maskWidth = convolutions.get(0).getWidth();
        	int inputHeight = input.get(0).getHeight();
        	int inputWidth = input.get(0).getWidth();
        	output = new ArrayList<>();
        	for (int i = 0; i < convolutions.size() / numChannels; i++) {
        		output.add(new Plate(new double[inputHeight - maskHeight + 1][inputWidth - maskWidth + 1]));
        	}
        } else {
        	for (int i = 0; i < output.size(); i++) {
        		Util.clear(output.get(i).getValues());
        	}
        }

        // Convolve each input with each mask.
        for (int i = 0; i < convolutions.size(); i += numChannels) {
            // convolve each input image, sum the output, add the new plate
            for (int j = 0; j < numChannels; j++) {
            	Plate mask = convolutions.get(i + j);
            	if (currentlyTraining) {
            		mask = new Plate(Util.scalarMultiply(1 - dropoutRate, mask.getValues(), false));
            	}
                Util.tensorAdd(
                		output.get(i / numChannels).getValues(), 
                		input.get(j).convolve(mask, activeNodes.get(i + j)).getValues(),
                		true);
            }
            output.get(i / numChannels).applyActivation(ActivationFunction.RELU);
        }
        previousOutput = deepCopyPlates(output);
        return output;
    }


    @Override
    public List<Plate> propagateError(List<Plate> errors, double learningRate) {
        if (errors.size() != previousOutput.size() || previousInput.isEmpty()) {
            throw new IllegalArgumentException("Bad propagation state.");
        }
        
        if (update == null) {
        	update = new double[convolutions.get(0).getHeight()][convolutions.get(0).getWidth()];
        }
        // hopefully these are static
        int errorHeight = errors.get(0).getHeight();
        int errorWidth = errors.get(0).getWidth();
        int convolutionHeight = convolutions.get(0).getHeight();
        int convolutionWidth = convolutions.get(0).getWidth();
        // Update the convolution values
        for (int i = 0; i < previousInput.size(); i++) {
        	Plate previous = previousInput.get(i);
            // Loop over the plate
            for (int j = 0; j <= errorHeight - convolutionHeight; j++) {
                for (int k = 0; k <= errorWidth - convolutionWidth; k++) {
                    for (int l = 0; l < convolutionHeight; l++) {
                        for (int m = 0; l < convolutionWidth; l++) {
                        	if (!activeNodes.get(i)[l][m]) {
                            	continue;
                            } else if (update[l][m] == 0) {
                                update[l][m] = 1;
                            }
                            update[l][m] = previous.valueAt(j + l, k + m)
                                    * errors.get(i).valueAt(j + l, k + m)
                                    * learningRate;
                        }
                    }
                    tensorAdd(convolutions.get(i).getValues(), update, true);
                }
            }
        }

        // Stores the delta values for all the plates in current layer
        if (deltaOutput == null) {
        	deltaOutput = new ArrayList<>(errors.size());
        	for (int i = 0; i < errors.size(); i++) {
        		deltaOutput.add(new Plate(new double[previousInput.get(0).getHeight()][previousInput.get(0).getWidth()]));
        	}
        } else {
        	for (int i = 0; i < deltaOutput.size(); i++) {
        		Util.clear(deltaOutput.get(i).getValues());
        	}
        }

        if (previousInput.size() == errors.size()) {
            for (int i = 0; i < errors.size(); i++) {
            	double[][] delta = deltaOutput.get(i).getValues();
            	Plate rotatedConvolution = convolutions.get(i).rot180();
                for (int row = 0; row <= delta.length - convolutions.get(i).getHeight(); row++) {
                    for (int col = 0; col <= delta[row].length - convolutions.get(i).getWidth(); col++) {
                        for (int kernelRow = 0; kernelRow < convolutions.get(i).getHeight(); kernelRow++) {
                            for (int kernelCol = 0; kernelCol < convolutions.get(i).getWidth(); kernelCol++) {
                            	if (!activeNodes.get(i)[convolutions.get(i).getHeight() - kernelRow - 1]
                            						   [convolutions.get(i).getWidth() - kernelCol - 1]) {
                            		continue;
                            	}
                                delta[row + kernelRow][col + kernelCol] += errors.get(i).valueAt(row, col)
                                        * rotatedConvolution.valueAt(kernelRow, kernelCol);
                            }
                        }
                    }
                }

                for (int row = 0; row < delta.length; row++) {
                    for (int col = 0; col < delta[row].length; col++) {
                        delta[row][col] *= ActivationFunction.RELU.applyDerivative(previousInput.get(i).valueAt(row, col));
                    }
                }
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
    
    private void determineDroppedOutNodes() {
    	resetDroppedOutNodes();
    	for (boolean[][] activeMatrix : activeNodes) {
    		int dropoutCount = 0;
        	int maxDropouts = activeMatrix.length * activeMatrix[0].length / 2;
    		for (int i = 0; i < activeMatrix.length; i++) {
    			for (int j = 0; j < activeMatrix[i].length; j++) {
    				if (dropoutRate > Util.RNG.nextDouble() && dropoutCount < maxDropouts) {
    					// The node "drops out."
    					activeMatrix[i][j] = false;
    					dropoutCount++;
    				}
    			}
    		}
    	}
    }
    
    private void resetDroppedOutNodes() {
    	for (boolean[][] activeMatrix : activeNodes) {
    		for (int i = 0; i < activeMatrix.length; i++) {
    			for (int j = 0; j < activeMatrix.length; j++) {
    				activeMatrix[i][j] = true;
    			}
    		}
    	}
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
        return "\n------\tConvolution Layer\t------\n\n" +
                String.format(
                        "Convolution Size: %dx%dx%d\n",
                        numChannels,
                        convolutions.get(0).getHeight(),
                        convolutions.get(0).getWidth()) +
                String.format(
                		"Output size     : %dx%dx%d\n",
                		numChannels,
                		outputHeight,
                		outputWidth) +
                String.format("Number of convolutions: %d\n", convolutions.size()) +
                "Activation Function: RELU\n" +
                String.format("Dropout rate: %.2f\n", dropoutRate) +
                "\n\t------------\t\n";
    }

    /**
     * Returns a new builder.
     */
    static Builder newBuilder() {
        return new Builder();
    }

    /**
     * A simple builder pattern for managing the layer's parameters at construction.
     */
    static class Builder {
        private int numChannels = 0;
        private int convolutionHeight = 0;
        private int convolutionWidth = 0;
        private int numConvolutions = 0;
        private double dropoutRate = 0;

        private Builder() {
        }

        Builder setConvolutionSize(int numChannels, int height, int width) {
        	if (!Lab3.RUNNING) {
        		checkPositive(numChannels, "Convolution channels", false);
        		checkPositive(height, "Convolution height", false);
        		checkPositive(width, "Convolution width", false);
        	}
            this.numChannels = numChannels;
            this.convolutionHeight = height;
            this.convolutionWidth = width;
            return this;
        }

        Builder setNumConvolutions(int numConvolutions) {
        	if (!Lab3.RUNNING) {
        		checkPositive(numConvolutions, "Number of convolutions", false);
        	}
            this.numConvolutions = numConvolutions;
            return this;
        }
        
        Builder setDropoutRate(double dropoutRate) {
        	if (dropoutRate < 0 || dropoutRate > 1) {
        		throw new IllegalArgumentException(
        				String.format("Dropout rate of %.2f is not valid.\n", dropoutRate));
        	}
        	this.dropoutRate = dropoutRate;
        	return this;
        }

        ConvolutionLayer build() {
        	if (!Lab3.RUNNING) {
        		checkPositive(numChannels, "Convolution channels", true);
        		checkPositive(convolutionHeight, "Convolution height", true);
        		checkPositive(convolutionWidth, "Convolution width", true);
        		checkPositive(numConvolutions, "Number of convolutions", true);
        	}
            List<Plate> convolutions = new ArrayList<>();
            for (int i = 0; i < numConvolutions; i++) {
                for (int j = 0; j < numChannels; j++) {
                    convolutions.add(
                            new Plate(
                                    createRandomConvolution(convolutionHeight, convolutionWidth)));
                }
            }
            return new ConvolutionLayer(convolutions, numChannels, dropoutRate);
        }

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
