package v2;

import static v2.Util.checkPositive;
import static v2.Util.checkValueInRange;

import java.util.ArrayList;
import java.util.List;

/**
 * A plate layer that performs max pooling on a specified window. There is no overlap between different placements of
 * the window.
 *
 * TODO: This implementation does not let you vary stride. In the future, we may want to add that feature.
 */
public class PoolingLayer implements PlateLayer {
    private final int windowHeight;
    private final int windowWidth;
    // quite similar to a plate, except its booleans so more memory efficient
    private ArrayList<boolean[][]> maximumOfWindow;
    private int numOutputs;
    private int outputHeight;
    private int outputWidth;

    private PoolingLayer(int numWindows, int windowHeight, int windowWidth) {
    	maximumOfWindow = new ArrayList<>(numWindows);
    	for (int i = 0; i < numWindows; i++) {
    		maximumOfWindow.add(null);
    	}
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    @Override
    public int getSize() {
    	return maximumOfWindow.size();
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
        // TODO: Reuse memory.
        if (maximumOfWindow.get(0) == null) {
            maximumOfWindow = new ArrayList<>();
            for (int j = 0; j < input.size(); j++) {
                maximumOfWindow.add(new boolean[input.get(j).getHeight()][input.get(j).getWidth()]);
            }
        }

        List<Plate> output = new ArrayList<Plate>(input.size());
        for (int i = 0; i < input.size(); i++) {
            output.add(maxPool(input.get(i), maximumOfWindow.get(i), windowHeight, windowWidth));
        }
        return output;
    }

    @Override
    public List<Plate> propagateError(List<Plate> gradients, double learningRate) {
        // TODO: Reuse memory.
        List<Plate> output = new ArrayList<>(gradients.size());
        for (int i = 0; i < gradients.size(); i++) {
            Plate errorPlate = gradients.get(i);
            double[][] upscaledValues = new double[maximumOfWindow.get(0).length][maximumOfWindow.get(0)[0].length];
            boolean[][] maximumOfPlate = maximumOfWindow.get(i);
            for (int j = 0; j < maximumOfPlate.length; j++) {
                for (int k = 0; k < maximumOfPlate[j].length; k++) {
                    // gradient is either copied from upper layer or zero - Ran Manor's answer at
                    // https://www.quora.com/In-neural-networks-how-does-backpropagation-get-carried-through-maxpool-layers
                    upscaledValues[j][k] = maximumOfPlate[j][k]
                            ? errorPlate.valueAt(j / windowHeight, k / windowWidth)
                            : 0;
                }
            }
            output.add(new Plate(upscaledValues));
        }
        return output;
    }

    @Override
    public void saveState() { /* Do nothing!*/ }
    
    @Override
    public void restoreState() { /* Do nothing! */ }

    /** Returns the max-pooled plate. No overlap between each pool. */
    public Plate maxPool(Plate plate, boolean[][] maximumOfPlate, int windowHeight, int windowWidth) {
        checkValueInRange(windowHeight, 0, plate.getHeight(), "Max pool window height");
        checkValueInRange(windowWidth, 0, plate.getWidth(), "Max pool window width");
        int resultHeight = plate.getHeight() / windowHeight;
        int resultWidth = plate.getWidth() / windowWidth;
        resultHeight += plate.getHeight() % windowHeight == 0 ? 0 : 1;
        resultWidth += plate.getWidth() % windowWidth == 0 ? 0 : 1;

        double[][] result = new double[resultHeight][resultWidth];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                int windowStartI = Math.min(i * windowHeight, plate.getHeight() - 1);
                int windowStartJ = Math.min(j * windowWidth, plate.getWidth() - 1);
                result[i][j] =
                        maxValInWindow(
                                plate,
                                maximumOfPlate,
                                windowStartI,
                                windowStartJ,
                                windowHeight,
                                windowWidth);
            }
        }
        return new Plate(result);
    }

    private double maxValInWindow(
            Plate plate, boolean[][] maximumOfPlate, int windowStartI, int windowStartJ, int windowHeight, int windowWidth) {
        double max = Double.MIN_VALUE;
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
        StringBuilder builder = new StringBuilder();
        builder.append("\n------\tPooling Layer\t------\n\n");
        builder.append(String.format("Window height: %d\n", windowHeight));
        builder.append(String.format("Window width: %d\n", windowWidth));
        builder.append("\n\t------------\t\n");
        return builder.toString();
    }

    /** Returns a new builder. */
    public static Builder newBuilder() {
        return new Builder();
    }

    /**
     * Simple builder pattern for creating PoolingLayers. (Not very helpful now, but later we may want to add other
     * parameters.)
     */
    public static class Builder {
        private int windowHeight = 0;
        private int windowWidth = 0;
        private int numWindows = 0;
        
        private Builder() {
        }

        public Builder setWindowSize(int height, int width) {
            checkPositive(height, "Window height", false);
            checkPositive(width, "Window width", false);
            this.windowHeight = height;
            this.windowWidth = width;
            return this;
        }
        
        public Builder setNumWindows(int numWindows) {
        	this.numWindows = numWindows;
        	return this;
        }

        public PoolingLayer build() {
            checkPositive(windowHeight, "Window height", true);
            checkPositive(windowWidth, "Window width", true);
            return new PoolingLayer(numWindows, windowHeight, windowWidth);
        }
    }
}
