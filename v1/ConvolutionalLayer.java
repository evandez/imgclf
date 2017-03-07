package v1;
public class ConvolutionalLayer extends Layer {

	public double[][][][] convolutions;
	public int stride;

	public ConvolutionalLayer(int numPlates, int imageDepth, int stride) {
		super(numPlates, imageDepth, stride, "conv");
		convolutions = new double[numPlates][imageDepth][stride][stride];
		plates = new DeepNode[numPlates][imageDepth][][];
		weights = new double[numPlates][imageDepth][][];
		for (int i = 0; i < numPlates; i++) {
			for (int j = 0; j < imageDepth; j++) {
				Lab3.randomizeWeights(convolutions[i][j]);
			}
		}
	}

	@Override
	public void connectPrev(Layer prev) {
		this.prev = prev;
		imageSize = prev.imageSize - stride + 1;
		
		for (int i = 0; i < numPlates; i++) {
			for (int j = 0; j < imageDepth; j++) {
				plates[i][j] = new DeepNode[imageSize][imageSize];
				convolutions[i][j] = new double[prev.imageSize][prev.imageSize];
				Lab3.randomizeWeights(weights[i][j]);
			}
		}
		// initialize nodes and connect to parents
		for (int i = 0; i < plates.length; i++) {
			for (int j = 0; j < plates[i].length; j++) {
				for (int k = 0; k < plates[i][j].length; k++) {
					for (int l = 0; l < plates[i][j][k].length; l++) {
						DeepNode newNode = new DeepNode(5, numPlates, imageDepth, stride);
						plates[i][j][k][l] = newNode;
						nodes.add(newNode);
						
						// connect previous layer to this node
						for (int m = 0; m < prev.plates.length; m++) {
							for (int n = 0; n < prev.plates[m].length; n++) {
								for (int o = 0; o < prev.plates[m][n].length; o++) {
									for (int p = 0; p < prev.plates[m][n][o].length; p++) {
										newNode.parents[m][n][o][p] = prev.plates[m][n][o][p];
									}
								}
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void connectNext(Layer next) {
		this.next = next;
	}

	@Override
	public void computeOutput() {
		for (int i = 0; i < nodes.size(); i++) {
			nodes.get(i).computeOutput();
		}
		next.computeOutput();
	}

	@Override
	public String toString() {
		return "conv - numPlates :" + numPlates + " stride: " + stride + " size:" + imageSize + "x" + imageSize;
	}

}
