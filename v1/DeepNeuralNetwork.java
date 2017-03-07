package v1;
public class DeepNeuralNetwork {

	public Layer[] layers;
	public int numPlates;
	public int imageSize;
	public int imageDepth;
	public int numCategories;
	// ex config:
	public DeepNeuralNetwork(String[] config, int numPlates, int imageSize, int imageDepth, int numCategories) {
		this.numPlates = numPlates;
		this.imageSize = imageSize;
		this.imageDepth = imageDepth;
		this.numCategories = numCategories;
		layers = new Layer[config.length];
		System.out.println("\nStructure: ");
		for (int i = 0; i < config.length; i++) {
			String layer = config[i].substring(0, 4);
			int param = 0;
			if (config[i].length() > 5) {
				param = Integer.parseInt(config[i].substring(5));
			}
			
			switch (layer) {
			case "conv":
				layers[i] = new ConvolutionalLayer(imageDepth, numPlates, param);
				break;
			case "pool":
				layers[i] = new PoolingLayer(imageDepth, numPlates, param);
				break;
			case "hidd":
				layers[i] = new HiddenLayer(imageDepth, param);
				break;
			case "inpu":
				layers[i] = new InputLayer(imageDepth, imageSize);
				break;
			case "outp":
				layers[i] = new OutputLayer(imageDepth, numCategories);
				break;
			default:
				throw new RuntimeException("Undefined layer type in config!");
			}
			if (i == 0) {
				System.out.println(layers[0]);
			}
			if (i > 0) {
				layers[i-1].connectNext(layers[i]);
				layers[i].connectPrev(layers[i - 1]);
				System.out.println(layers[i]);
			}
		}
		System.out.println();
	}
	
	public void setInput(Instance inst) {
		int[][][] intImage = new int[][][] {
			inst.getRedChannel(),
			inst.getGreenChannel(),
			inst.getBlueChannel(),
			inst.getGrayImage()
		};
		
		double[][][] image = new double[intImage.length][intImage[0].length][intImage[0][0].length];
		for (int i = 0; i < intImage.length; i++) {
			for (int j = 0; j < intImage[i].length; j++) {
				for (int k = 0; k < intImage[i][j].length; k++) {
					image[i][j][k] = intImage[i][j][k] / 255.0;
				}
			}
		}
		
		((InputLayer)layers[0]).setInput(image);
	}
	
	public void computeOutput() {
		layers[0].computeOutput();
	}
}
