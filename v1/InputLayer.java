package v1;
public class InputLayer extends Layer {

	public int imageDepth;

	public InputLayer(int imageDepth, int imageSize) {
		super(1, imageDepth, 1, "inpu");
		this.imageSize = imageSize;
		// initialize plates and input nodes
		for (int i = 0; i < imageDepth; i++) {
			plates[0][i] = new DeepNode[imageSize][imageSize];
			for (int j = 0; j < imageSize; j++) {
				for (int k = 0; k < imageSize; k++) {
					DeepNode newNode = new DeepNode(0, 1, imageDepth, 0);
					plates[0][i][j][k] = newNode;
					nodes.add(newNode);
				}
			}
		}
	}

	@Override
	public void connectPrev(Layer prev) {
		throw new RuntimeException("Cannot connect a previous layer to an input layer!");
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
	
	public void setInput(double[][][] image) {
		for (int i = 0; i < image.length; i++) { 
			for (int j = 0; j < image[i].length; j++) {
				for (int k = 0; k < image[i][j].length; k++) {
					plates[0][i][j][k].setInput(image[i][j][k]);
				}
			}
		}
	}

	@Override
	public String toString() {
		return "inpu - imageSize :" + imageSize + "x" + imageSize + " imageDepth:" + super.imageDepth;
	}

}
