import java.util.ArrayList;

public abstract class Layer {

	public Layer next, prev;

	public int numPlates;
	public int imageDepth;
	public int imageSize;
	public int stride;

	public String type;

	// plates[plateIndex][channelIndex][height][width] where plateIndex is the index of the plate, channelIndex is 0 for
	// red, 1 for green, 2 for blue, 3 for gray (note that Jude said gray channel is pretty broken)
	public DeepNode[][][][] plates;
	// indexing is the same
	public double[][][][] weights;

	public ArrayList<DeepNode> nodes;

	public Layer(int numPlates, int imageDepth, int stride, String type) {
		this.numPlates = numPlates;
		this.imageDepth = imageDepth;
		this.stride = stride;
		this.type = type;
		this.plates = new DeepNode[numPlates][imageDepth][][];
		this.weights = new double[numPlates][imageDepth][][];
		this.nodes = new ArrayList<>();
	}

	public abstract void connectPrev(Layer prev);

	public abstract void connectNext(Layer next);

	public abstract void computeOutput();

	public abstract String toString();
}
