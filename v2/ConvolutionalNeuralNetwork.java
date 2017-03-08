package v2;

import static v2.Util.checkNotEmpty;
import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A convolutional neural network that supports arbitrary convolutional and pooling layers,
 * followed by arbitrarily many fully-connected layers.
 */
public class ConvolutionalNeuralNetwork {
	private final int inputHeight;
	private final int inputWidth;
	private final List<PlateLayer> plateLayers;
	private final List<FullyConnectedLayer> fullyConnectedLayers;
	private final List<String> classes;
	private final int maxEpochs;
	private final double learningRate;
	private final boolean useRGB;

	private ConvolutionalNeuralNetwork(
			int inputHeight,
			int inputWidth,
			List<PlateLayer> plateLayers,
			List<FullyConnectedLayer> fullyConnectedLayers,
			List<String> classes,
			int maxEpochs,
			double learningRate,
			boolean useRGB) {
		this.inputHeight = inputHeight;
		this.inputWidth = inputWidth;
		this.plateLayers = plateLayers;
		this.fullyConnectedLayers = fullyConnectedLayers;
		this.classes = classes;
		this.maxEpochs = maxEpochs;
		this.learningRate = learningRate;
		this.useRGB = useRGB;
	}
	
	/** Trains the CNN with the given training data and tuning data. */
	public void train(Dataset trainSet, Dataset tuneSet, boolean verbose) {
		double prevAccuracy = 0.0;
		double currAccuracy = 0.0;
		for (int epoch = 0; epoch < maxEpochs; epoch++) {
			trainSingleEpoch(trainSet);
			currAccuracy = test(tuneSet, false);
			
			if (verbose) {
				System.out.printf("Epoch %d completed with tune accuracy of %.5f\n", epoch, currAccuracy);
			}

			if (currAccuracy < prevAccuracy) {
				break;
			}
		}
	}

	/** Passes all images in the dataset through the network and backpropagates the errors. */
	private void trainSingleEpoch(Dataset trainSet) {
		for (Instance img : trainSet.getImages()) {
			double[] output = computeOutput(img);
			
			// First, propagate error through fully connected layers.
			double[] fcError = labelToOneOfN(img.getLabel());
			for (int i = 0; i < fcError.length; i++) {
				fcError[i] -= output[i];
			}
			
			for (FullyConnectedLayer layer : fullyConnectedLayers) {
				fcError = layer.propagateError(fcError, learningRate);
			}
			
			// Then, propagate error through plate layers.
			List<Plate> plateErrors = unpackPlates(fcError);
			for (PlateLayer layer : plateLayers) {
				plateErrors = layer.propagateError(plateErrors, learningRate);
			}
		}
	}
	
	/**
	 * Returns the prediction accuracy of this classifier on the test set.
	 * 
	 * Here, accuracy is numCorrectlyClassified/numExamples.
	 */
	public double test(Dataset testSet, boolean verbose) {
		int errCount = 0;
		for (Instance img : testSet.getImages()) {
			String predicted = classify(img);
			if (!predicted.equals(img.getLabel())) {
				errCount++;
			}
			
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s\n", predicted, img.getLabel());
			}
		}
		return ((double) (testSet.getSize() - errCount)) / testSet.getSize();
	}
	
	/** Returns the predicted label for the image. */
	public String classify(Instance img) {
		double[] probs = computeOutput(img);
		double maxProb = -1;
		int bestIndex = -1;
		for (int i = 0; i < probs.length; i++) {
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				bestIndex = i;
			}
		}
		return classes.get(bestIndex);
	}
	
	/**
	 * Propagates the image through the network and returns the last
	 * (fully-connected) layer's output.
	 */
	private double[] computeOutput(Instance img) {
		// Pass the input through the plate layers first.
		List<Plate> plates = Arrays.asList(instanceToPlate(img));
		for (PlateLayer layer : plateLayers) {
			plates = layer.computeOutput(plates);
		}
		
		// Then pass the output through the fully connected layers.
		double[] vec = packPlates(plates);
		for (FullyConnectedLayer fcLayer : fullyConnectedLayers) {
			vec = fcLayer.computeOutput(vec);
		}
		return vec;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("\n//////\tNETWORK SPECIFICATIONS\t//////\n");
		builder.append(String.format("Input Height: %d\n", inputHeight));
		builder.append(String.format("Input Width: %d\n", inputWidth));
		builder.append(String.format("Number of plate layers: %d\n", plateLayers.size()));
		builder.append(
				String.format(
						"Number of fully connected hidden layers: %d\n",
						fullyConnectedLayers.size() - 1));
		builder.append(
				String.format("Predicts these classes: %s\n", classes));
		builder.append(String.format("Using RGB: %b\n", useRGB));
		builder.append("\n//////\tNETWORK STRUCTURE\t//////\n");
		if (plateLayers.isEmpty()) {
			builder.append("\n------\tNo plate layers!\t------\n");
		} else {
			for (PlateLayer plateLayer : plateLayers) {
				builder.append(plateLayer.toString());
			}
		}
		for (FullyConnectedLayer fcLayer : fullyConnectedLayers) {
			builder.append(fcLayer.toString());
		}
		return builder.toString();
	}
	
	private double[] labelToOneOfN(String label) {
		double[] correctOutput = new double[classes.size()];
		correctOutput[classes.indexOf(label)] = 1;
		return correctOutput;
	}
	
	private Plate[] instanceToPlate(Instance instance) {
		if (useRGB) {
			return new Plate[] {
					new Plate(intImgToDoubleImg(instance.getRedChannel())),
					new Plate(intImgToDoubleImg(instance.getBlueChannel())),
					new Plate(intImgToDoubleImg(instance.getGreenChannel())),
					new Plate(intImgToDoubleImg(instance.getGrayImage())),
			};
		} else {
			return new Plate[] {new Plate(intImgToDoubleImg(instance.getGrayImage()))};
		}
	}
	
	private static double[][] intImgToDoubleImg(int[][] intImg) {
		double[][] dblImg = new double[intImg.length][intImg[0].length];
		for (int i = 0; i < dblImg.length; i++) {
			for (int j = 0; j < dblImg[i].length; j++) {
				dblImg[i][j] = (255 - intImg[i][j]) / 255;
			}
		}
		return dblImg;
	}
	
	/** 
	 * Pack the plates into a single, 1D double array. Used to connect the plate layers
	 * with the fully connected layers.
	 */
	private static double[] packPlates(List<Plate> plates) {
		checkNotEmpty(plates, "Plates to pack", false);
		int flattenedPlateSize = plates.get(0).getTotalNumValues();
		double[] result = new double[flattenedPlateSize * plates.size()];
		for (int i = 0; i < plates.size(); i++) {
			System.arraycopy(
					plates.get(i).as1DArray(),
					0 /* Copy the whole flattened plate! */,
					result,
					i * flattenedPlateSize,
					flattenedPlateSize);
		}
		return result;
	}
	
	/** Unpack the 1D double array into a list of plates (3D double tensors). */
	private static List<Plate> unpackPlates(double[] packedPlates) {
		// TODO: Implement this method.
        List<Plate> plates = new ArrayList<>();
        for (int i = 0; i < packedPlates.length; i++) {
            // TODO: Need to get the size of each plate to create list

        }
        return plates;
	}
	
	/** Returns a new builder. */
	public static Builder newBuilder() { return new Builder(); }
	
	/** A builder pattern for managing the many parameters of the network. */
	public static class Builder {
		private final List<PlateLayer> plateLayers = new ArrayList<>();
		private List<String> classes = null;
		private int inputHeight = 0;
		private int inputWidth = 0;
		private int fullyConnectedWidth = 0;
		private int fullyConnectedDepth = 0;
		private int maxEpochs = 0;
		private double learningRate = 0;
		private boolean useRGB = true;
		
		private Builder() {}
		
		public Builder setInputHeight(int height) {
			checkPositive(height, "Input height", false);
			this.inputHeight = height;
			return this;
		}
		
		public Builder setInputWidth(int width) {
			checkPositive(width, "Input width", false);
			this.inputWidth = width;
			return this;
		}
		
		public Builder appendConvolutionLayer(ConvolutionLayer layer) {
			return appendPlateLayer(layer);
		}
		
		public Builder appendPoolingLayer(PoolingLayer layer) {
			return appendPlateLayer(layer);
		}
		
		private Builder appendPlateLayer(PlateLayer layer) {
			checkNotNull(layer, "Plate layer");
			this.plateLayers.add(layer);
			return this;
		}
		
		public Builder setFullyConnectedWidth(int width) {
			checkPositive(width, "Fully connected width", false);
			this.fullyConnectedWidth = width;
			return this;
		}
		
		public Builder setFullyConnectedDepth(int depth) {
			checkPositive(depth, "Fully connected depth", false);
			this.fullyConnectedDepth = depth;
			return this;
		}
		
		public Builder setClasses(List<String> classes) {
			checkNotNull(classes, "Classes");
			checkNotEmpty(classes, "Classes", false);
			this.classes = classes;
			return this;
		}
		
		public Builder setMaxEpochs(int maxEpochs) {
			checkPositive(maxEpochs, "Max epochs", false);
			this.maxEpochs = maxEpochs;
			return this;
		}
		
		public Builder setLearningRate(double learningRate) {
			checkPositive(learningRate, "Learning rate", false);
			this.learningRate = learningRate;
			return this;
		}
		
		public Builder setUseRGB(boolean useRGB) {
			this.useRGB = useRGB;
			return this;
		}
		
		public ConvolutionalNeuralNetwork build() {
			// No check for nonemptyness of plate layers - if none provided, use fully connected.
			checkNotNull(classes, "Classes");
			checkPositive(inputHeight, "Input height", true);
			checkPositive(inputWidth, "Input width", true);
			checkPositive(fullyConnectedWidth, "Fully connected width", true);
			checkPositive(fullyConnectedDepth, "Fully connected depth", true);
			checkPositive(maxEpochs, "Max epochs", true);
			checkPositive(learningRate, "Learning rate", true);
			// No check for useRGB. Just default to true.

			// Given input dimensions, determine how many plates will be output by
			// the last plate layer, and the dimensions of those plates.
			// Note that if there are no plate layers, then this result defaults to
			// imageHeight * imageWidth, which is what we need in that case.
			int outputHeight = inputHeight;
			int outputWidth = inputWidth;
			int outputChannels = (useRGB) ? 4 : 1;
			int numOutputs = ((ConvolutionLayer)plateLayers.get(0)).numConvolutions();
			for (PlateLayer plateLayer : plateLayers) {
				outputHeight = plateLayer.calculateOutputHeight(outputHeight);
				outputWidth = plateLayer.calculateOutputWidth(outputWidth);
				numOutputs = plateLayer.calculateNumOutputs(numOutputs);
			}

			List<FullyConnectedLayer> fullyConnectedLayers = new ArrayList<>(fullyConnectedDepth);
			
			// Always have at least one hidden layer - add it first.
			// TODO: Make the fully-connected activation function a parameter.
			fullyConnectedLayers.add(FullyConnectedLayer.newBuilder()
					.setActivationFunction(ActivationFunction.RELU)
					.setNumInputs(outputWidth * outputHeight * numOutputs)
					.setNumNodes(fullyConnectedWidth)
					.build());
			
			// Add the other hidden layers.
			for (int i = 0; i < fullyConnectedDepth - 1; i++) {
				fullyConnectedLayers.add(FullyConnectedLayer.newBuilder()
						.setActivationFunction(ActivationFunction.RELU)
						.setNumInputs(fullyConnectedWidth)
						.setNumNodes(fullyConnectedWidth)
						.build());
			}

			// Add the output layer.
			fullyConnectedLayers.add(FullyConnectedLayer.newBuilder()
					.setActivationFunction(ActivationFunction.SIGMOID)
					.setNumInputs(fullyConnectedWidth)
					.setNumNodes(classes.size())
					.build());
			
			return new ConvolutionalNeuralNetwork(
					inputHeight,
					inputWidth,
					plateLayers,
					fullyConnectedLayers,
					classes,
					maxEpochs,
					learningRate,
					useRGB);
		}
	}
}
