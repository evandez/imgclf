/**
 * Class for internal organization of a deep Neural Network. There are 7 types of nodes.
 */
public class DeepNode {
	// 0 = input, 1 = biasToHidden, 2 = hidden, 3 = biasToOutput, 4 = Output, 5 = convolutional, 6 = pool
	private int type = 0;
	// Array List that will contain the parents (including the bias node) with weights if applicable
	public DeepNode[][][][] parents;
	public double[][][][] weights;

	public Double inputValue = 0.0;
	public Double outputValue = 0.0;
	private Double sum = 0.0; // sum of wi*xi
	public int numPlates;
	public int imageDepth;
	public int stride;

	// Create a node with a specific type
	public DeepNode(int type, int numPlates, int imageDepth, int stride) {
		this.numPlates = numPlates;
		this.imageDepth = imageDepth;
		this.stride = stride;
		if (type > 6 || type < 0) {
			throw new RuntimeException("Invalid node type!");
		} else {
			this.type = type;
		}

		if (this.type == 2 || this.type == 4 || this.type == 5 || this.type == 6) {
			parents = new DeepNode[numPlates][imageDepth][][];
			weights = new double[numPlates][imageDepth][][];
		}
	}

	// For an input node sets the input value which will be the value of a particular attribute
	public void setInput(Double inputValue) {
		if (type == 0) {
			this.inputValue = inputValue;
		} else {
			throw new RuntimeException("Not an input node!");
		}
	}

	/**
	 * Calculate the output of a ReLU node.
	 * 
	 * @param train:
	 *           the training set
	 */
	public double computeOutput() {
		sum = 0.0;
		if (type == 0 || type == 1) {
			return 0.0;
		} else if (type == 4) {
			sum = 0.0;
			for (int i = 0; i < parents[0][0][0].length; i++) {
				sum += parents[0][0][0][i].getOutput() * weights[0][0][0][i];
			}
			outputValue = sum;
			return outputValue;
		} else if (this.type == 2 || this.type == 5) {
			for (int i = 0; i < parents.length; i++) {
				for (int j = 0; j < parents[i].length; j++) {
					for (int k = 0; k < parents[i][j].length; k++) {
						for (int l = 0; l < parents[i][j][k].length; l++) {
							sum += weights[i][j][k][l] * parents[i][j][k][l].getOutput();
						}
					}
				}
			}
			outputValue = sum = Math.max(0.0, sum);
			return outputValue;
		} else if (this.type == 6) {
			for (int i = 0; i < parents.length; i++) {
				for (int j = 0; j < parents[i].length; j++) {
					for (int k = 0; k < parents[i][j].length; k++) {
						for (int l = 0; l < parents[i][j][k].length; l++) {
							sum = Math.max(sum, weights[i][j][k][l] * parents[i][j][k][l].getOutput());
						}
					}
				}
			}
			outputValue = sum;
			return outputValue;
		} else {
			throw new RuntimeException("Not an output or hidden node!");
		}
	}

	public double getSum() {
		return sum;
	}

	// Gets the output value
	public double getOutput() {
		if (type == 0) {
			return inputValue;
		} else if (type == 1 || type == 3) {
			return 1;
		} else {
			return outputValue;
		}
	}
}
