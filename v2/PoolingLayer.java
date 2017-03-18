package v2;

import static v2.Util.checkPositive;
import static v2.Util.checkValueInRange;

import java.util.ArrayList;
import java.util.List;

/**
 * A plate layer that performs max pooling on a specified window. There is no overlap between different placements of
 * the window.
 *
 */
public class PoolingLayer implements PlateLayer {
    private final int windowHeight;
    private final int windowWidth;
    // quite similar to a plate, except its booleans so more memory efficient
    private int numOutputs;
    private int outputHeight;
    private int outputWidth;
    private List<Plate> errors;
    private List<Plate> output;
    private double[][][] upscaledValues;
    private boolean[][][] maximumOfWindow;

    private PoolingLayer(int numWindows, int windowHeight, int windowWidth) {
    	maximumOfWindow = new boolean[numWindows][][];
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    @Override
    public int getSize() {
    	return maximumOfWindow.length;
    }
    
    @Override
    public int calculateNumOutputs(int numInputs) {
    	numOutputs = numInputs;
        return numOutputs;
    }
    
    @Override
    public int calculateNumOutputs() {
    	if (numOutputs > 0) {
    		return numOutputs;
    	} else {
    		throw new RuntimeException("Cannot call calculateNumOutputs without arguments before calling it with arguments.");
    	}
    }

    @Override
    public int calculateOutputHeight(int inputHeight) {
        outputHeight = inputHeight / windowHeight;
        if (inputHeight % windowHeight > 0) {
            outputHeight++;
        }
        return outputHeight;
    }
    
    @Override
    public int calculateOutputHeight() {
    	if (outputHeight > 0) {
    		return outputHeight;
    	} else {
    		throw new RuntimeException("Cannot call calculateOutputHeight without arguments before calling it with arguments.");
    	}
    }

    @Override
    public int calculateOutputWidth(int inputWidth) {
        outputWidth = inputWidth / windowWidth;
        if (inputWidth % windowWidth > 0) {
            outputWidth++;
        }
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
    public List<Plate> computeOutput(List<Plate> input) {
        if (maximumOfWindow[0] == null) {
            maximumOfWindow = new boolean[input.size()][input.get(0).getHeight()][input.get(0).getWidth()];
        }
        int resultHeight = input.get(0).getHeight() / windowHeight;
        int resultWidth = input.get(0).getWidth() / windowWidth;
        resultHeight += input.get(0).getHeight() % windowHeight == 0 ? 0 : 1;
        resultWidth += input.get(0).getWidth() % windowWidth == 0 ? 0 : 1;
        if (output == null) {
        	output = new ArrayList<>(input.size());
        	for (int i = 0; i < input.size(); i++) {
        		output.add(new Plate(new double[resultHeight][resultWidth]));
        	}
        }
        
        for (int i = 0; i < input.size(); i++) {
        	double[][] result = output.get(i).getValues();
        	  for (int j = 0; j < result.length; j++) {
                  for (int k = 0; k < result[j].length; k++) {
                      int windowStartI = Math.min(j * windowHeight, input.get(i).getHeight() - 1);
                      int windowStartJ = Math.min(k * windowWidth, input.get(i).getWidth() - 1);
                      result[j][k] =
                              maxValInWindow(
                                      input.get(i),
                                      maximumOfWindow[i],
                                      windowStartI,
                                      windowStartJ,
                                      windowHeight,
                                      windowWidth);
                  }
              }
        }
        return output;
    }

    @Override
    public List<Plate> propagateError(List<Plate> gradients, double learningRate) {
    	if (errors == null) {
    		errors = new ArrayList<>(gradients.size());
    		upscaledValues = new double[gradients.size()][maximumOfWindow[0].length][maximumOfWindow[0][0].length];
    		for (int i = 0; i < gradients.size(); i++) {
    			errors.add(new Plate(upscaledValues[i]));
    		}
    	}
        for (int i = 0; i < gradients.size(); i++) {
            Plate errorPlate = gradients.get(i);
            for (int j = 0; j < maximumOfWindow[i].length; j++) {
                for (int k = 0; k < maximumOfWindow[i][j].length; k++) {
                    upscaledValues[i][j][k] = maximumOfWindow[i][j][k]
                            ? errorPlate.valueAt(j / windowHeight, k / windowWidth)
                            : 0;
                }
            }
        }
        return errors;
    }

    @Override
    public void saveState() { /* Do nothing!*/ }
    
    @Override
    public void restoreState() { /* Do nothing! */ }

    private double maxValInWindow(
            Plate plate, boolean[][] maximumOfPlate, int windowStartI, int windowStartJ, int windowHeight, int windowWidth) {
        double max = -Double.MAX_VALUE; 
        int windowEndI = Math.min(windowStartI + windowHeight - 1, plate.getHeight() - 1);
        int windowEndJ = Math.min(windowStartJ + windowWidth - 1, plate.getWidth() - 1);
        int maxI = -1;
        int maxJ = -1;
        for (int i = windowStartI; i <= windowEndI; i++) {
            for (int j = windowStartJ; j <= windowEndJ; j++) {
                double value = plate.getValues()[i][j];
                if (value > max) {
                    max = value;
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        maximumOfPlate[maxI][maxJ] = true;
        return max;
    }

    @Override
    public String toString() {
        return  "\n------\tPooling Layer\t------\n\n" +
                String.format("Window size: %dx%d\n", windowHeight, windowWidth) +
                String.format("Output size: %dx%d\n", outputHeight, outputWidth) +
                "\n\t------------\t\n";
    }

    /** Returns a new builder. */
    static Builder newBuilder() {
        return new Builder();
    }

    /**
     * Simple builder pattern for creating PoolingLayers. (Not very helpful now, but later we may want to add other
     * parameters.)
     */
    static class Builder {
        private int windowHeight = 0;
        private int windowWidth = 0;
        private int numWindows = 0;
        
        private Builder() {
        }

        Builder setWindowSize(int height, int width) {
            checkPositive(height, "Window height", false);
            checkPositive(width, "Window width", false);
            this.windowHeight = height;
            this.windowWidth = width;
            return this;
        }
        
        Builder setNumWindows(int numWindows) {
        	this.numWindows = numWindows;
        	return this;
        }

        PoolingLayer build() {
            checkPositive(windowHeight, "Window height", true);
            checkPositive(windowWidth, "Window width", true);
            return new PoolingLayer(numWindows, windowHeight, windowWidth);
        }
    }
}
