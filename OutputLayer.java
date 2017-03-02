public class OutputLayer extends Layer {

	public int numOutputs;

	// for this layer type, the weights is is 1x1x1xnumHidden
	// same for parents
	public OutputLayer(int imageDepth, int numOutputs) {
		super(1, 1, 1, "outp");
		this.numOutputs = numOutputs;
		this.numPlates = 1;
		this.stride = 1;
	}

	@Override
	public void connectPrev(Layer prev) {
		imageSize = prev.imageSize;
		if (prev.type.equals("hidd")) {
			weights[0][0] = new double[0][prev.nodes.size()];
			Lab3.randomizeWeights(weights[0][0]);
			// initialize nodes and connect to parents
			for (int i = 0; i < numOutputs; i++) {
				DeepNode newNode = new DeepNode(4, 1, 1, 1);
				nodes.add(newNode);
				newNode.parents = new DeepNode[1][1][1][prev.nodes.size()];
				newNode.weights[0][0] = new double[1][((HiddenLayer) prev).numHidden];
				// add every hidden node as a prent of the new node
				// initialize the weight of each node-parent connection
				for (int j = 0; j < prev.nodes.size(); j++) {
					newNode.parents[0][0][0][j] = prev.nodes.get(j);
					newNode.weights[0][0][0][j] = Math.random() * Lab3.INITIAL_WEIGHT;
				}
			}
		} else {
			throw new RuntimeException(
					"Support for direct connectiosn to output layer without hidden layer not supported.");
		}
	}

	@Override
	public void connectNext(Layer next) {
		throw new RuntimeException("Cannot connect a next layer to an output layer!");
	}

	@Override
	public void computeOutput() {
		for (int i = 0; i < nodes.size(); i++) {
			nodes.get(i).computeOutput();
		}
	}

	@Override
	public String toString() {
		return "outp - numOutputs:" + nodes.size();
	}

}
