package v1;
public class HiddenLayer extends Layer{
	public int numHidden;
	
	public HiddenLayer(int imageDepth, int numHidden) {
		super(1, imageDepth, 1, "hidd");
		this.numHidden = numHidden;
		// hidden nodes
	}

	@Override
	public void connectPrev(Layer prev) {
		this.prev = prev;
		this.imageSize = prev.imageSize;
		// initialize nodes and connect to parents
		for (int i = 0; i < numHidden; i++) {
			DeepNode newNode = new DeepNode(2, 1, 1, 1);
			nodes.add(newNode);
			newNode.parents = new DeepNode[numPlates][imageDepth][imageSize][imageSize];
			for (int j = 0; j < weights.length; j++) {
				for (int k = 0; k < weights[j].length; k++) {
					weights[j][k] = new double[imageSize][imageSize];
					Lab3.randomizeWeights(weights[j][k]);
					for (int l = 0; l < weights[j][k].length; l++) {
						for (int m = 0; m < weights[j][k][l].length; m++) {
							newNode.parents[j][k][l][m] = prev.plates[j][k][l][m];
						}
					}
				}
			}
			newNode.weights = weights.clone();
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
		return "hidd - numNodes  :" + nodes.size();
	}
}
