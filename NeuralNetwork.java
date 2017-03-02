import java.util.ArrayList;
import java.util.Vector;

public class NeuralNetwork {
	private static final double WEIGHT = .0001;
	private static final double EPSILON = .0001;
	private static final int MAX_DECREASE = 25;
	private static final int HISTORY_LENGTH = 100;
	private static double learningRate;
	private static double momentumRate;
	private static double regularizationRate;

	private ArrayList<Node> inputNodes = null; // list of the output layer nodes.
	private ArrayList<Node> hiddenNodes = null; // list of the hidden layer nodes
	private ArrayList<Node> outputNodes = null; // list of the output layer nodes

	private double[] errorK;
	private double[] deltaK;
	private double[][] deltaJK;
	private double[][] deltaIJ;
	private double[][] weightsIJ;
	private double[][] weightsJK;

	// Weight updates from previous iterations.
	private double[][] momentumJK;
	private double[][] momentumIJ;

	/**
	 * This constructor creates the nodes necessary for the neural network and connects the nodes of different layers.
	 * After calling the constructor the last node of both inputNodes and hiddenNodes will be bias nodes.
	 */
	public NeuralNetwork(int numInputs, int numHiddens, int numOutputs) {
		// Reading the weights
		weightsIJ = new double[numInputs + 1][numHiddens];
		weightsJK = new double[numHiddens + 1][numOutputs];
		Lab3.randomizeWeights(weightsIJ);
		Lab3.randomizeWeights(weightsJK);
		System.out
				.println("Input nodes: " + numInputs + "\nHidden nodes: " + numHiddens + "\nOutput nodes: " + numOutputs);
		// input layer nodes
		inputNodes = new ArrayList<Node>();
		for (int i = 0; i < numInputs; i++) {
			inputNodes.add(new Node(0));
		}
		// bias node from input layer to hidden
		inputNodes.add(new Node(1));

		// hidden layer nodes
		hiddenNodes = new ArrayList<Node>();
		for (int j = 0; j < numHiddens; j++) {
			hiddenNodes.add(new Node(2));
			// fully connecting hidden layer nodes with input layer nodes
			for (int i = 0; i < inputNodes.size(); i++) {
				hiddenNodes.get(j).parents.add(new NodeWeightPair(inputNodes.get(i), weightsIJ[i][j]));
			}
		}
		// bias node from hidden layer to output
		hiddenNodes.add(new Node(3));

		// Output node layer
		outputNodes = new ArrayList<Node>();
		for (int k = 0; k < numOutputs; k++) {
			outputNodes.add(new Node(4));
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				outputNodes.get(k).parents.add(new NodeWeightPair(hiddenNodes.get(j), weightsJK[j][k]));
			}
		}

		errorK = new double[outputNodes.size()];

		deltaK = new double[outputNodes.size()];
		momentumJK = new double[hiddenNodes.size()][outputNodes.size()];
		deltaJK = new double[hiddenNodes.size()][outputNodes.size()];
		momentumIJ = new double[inputNodes.size()][hiddenNodes.size() - 1];
		deltaIJ = new double[inputNodes.size()][hiddenNodes.size() - 1];
	}

	public static void setParameters(double learningRate, double momentumRate, double regularizationRate) {
		NeuralNetwork.learningRate = learningRate;
		NeuralNetwork.momentumRate = momentumRate;
		NeuralNetwork.regularizationRate = regularizationRate;
	}

	public double accuracy(Vector<Vector<Double>> test, boolean verbose) {
		int correct = 0;

		for (int i = 0; i < test.size(); i++) {
			// getting output from network
			int outputClassValue = calculateOutputForInstance(test.get(i), verbose);

			int actualClassValue = test.get(i).get(Lab3.inputVectorSize - 1).intValue();

			if (actualClassValue == -1) {
				throw new RuntimeException("Bad class value!");
			} else if (outputClassValue == actualClassValue) {
				correct++;
			}
		}
		return (double) correct / test.size();
	}

	/** Get the output (index of class value) from the neural network for a single instance. */
	public int calculateOutputForInstance(Vector<Double> inst, boolean verbose) {
		int j = 0;
		for (int i = 0; i < inputNodes.size() - 1; i++) {
			inputNodes.get(i).setInput(inst.get(i));
		}
		for (int i = 0; i < hiddenNodes.size() - 1; i++) {
			hiddenNodes.get(i).computeOutput();
		}

		for (int i = 0; i < outputNodes.size(); i++) {
			outputNodes.get(i).computeOutput();
		}

//		if (verbose && Math.random() > 1) {
//			System.out.println("header:\n" + inputNodes.get(0).getOutput() + " " + inputNodes.get(10).getOutput() + " "
//					+ inputNodes.get(20).getOutput() + " " + hiddenNodes.get(0).getOutput() + " "
//					+ hiddenNodes.get(1).getOutput() + " " + hiddenNodes.get(2).getOutput() + " "
//					+ outputNodes.get(0).getOutput() + " " + outputNodes.get(1).getOutput() + " "
//					+ outputNodes.get(2).getOutput());
//		}

		double maxOutput = -1;
		int outputIndex = -1; // index of max output
		// find max output rounded to tenths
		for (int i = 0; i < outputNodes.size(); i++) {
			if (verbose) {
				System.out.printf("%.15f", outputNodes.get(i).getOutput());
				System.out.print("   ");
			}
			if (outputNodes.get(i).getOutput() > maxOutput) {
				maxOutput = outputNodes.get(i).getOutput();
				outputIndex = i;
			}
		}

		if (verbose) {
			double clazz = inst.get(inst.size() - 1);

			System.out.print("    output: " + outputIndex + " vs actual: " + clazz);

			if (clazz != outputIndex) {
				System.out.print("   x");
			}

			System.out.println();
		}

		return outputIndex;
	}

	/** Calculate errors, collect weights, and calculate deltas. */
	public void updateWeights(Vector<Double> inst) {
		calculateErrors(inst);
		calculateMomentum();
		calculateDeltas();
		applyDeltas();
	}

	/**
	 * Calculate the error of each output node.
	 */
	private void calculateErrors(Vector<Double> inst) {
		calculateOutputForInstance(inst, false); // calculate output node outputs

		int label = inst.get(inst.size() - 1).intValue();
//		boolean print = Math.random() > 1;
		for (int k = 0; k < outputNodes.size(); k++) // calculate error for each output node
		{
			double out = outputNodes.get(k).getOutput();
//			System.out.print("out: " + out + " ");
			if (k == label) {
//				if (print) {
//					System.out.print(" k: ");
//				}
//				errorK[k] = (1 - out) * out * (1 - out);
				if (out > 0) {
					errorK[k] = 1 - out;
				} else {
					errorK[k] = -.000001;
				}
			} else {
//				errorK[k] = (0 - out) * out * (1 - out);
				if (out > 0) {
					errorK[k] = 0 - out;
				} else {
					errorK[k] = -.000001;
				}
			}
//			if (print) {
//				System.out.print(errorK[k] + " . ");
//			}
		}
//		if (print) {
//			System.out.println();
//		}
	}

	/** Calculate momentum terms. */
	private void calculateMomentum() {
		if (deltaJK == null) {
			momentumJK = new double[hiddenNodes.size()][outputNodes.size()];
		} else {
			arrayCopy2D(deltaJK, momentumJK);
			for (int j = 0; j < momentumJK.length; j++) {
				for (int k = 0; k < momentumJK[j].length; k++) {
					momentumJK[j][k] *= momentumRate;
				}
			}
		}

		if (deltaIJ == null) {
			momentumIJ = new double[inputNodes.size()][hiddenNodes.size() - 1];
		} else {
			arrayCopy2D(deltaIJ, momentumIJ);
			for (int i = 0; i < momentumIJ.length; i++) {
				for (int j = 0; j < momentumIJ[i].length; j++) {
					momentumIJ[i][j] *= momentumRate;
				}
			}
		}
	}

	/** Calculate the change of each arc and save into global variables. */
	private void calculateDeltas() {

		// int jkZero = 0;
		// int count = 0;
		for (int k = 0; k < outputNodes.size(); k++) {
			if (outputNodes.get(k).getSum() < 0) {
				continue;
			}

			deltaK[k] = errorK[k];

			ArrayList<NodeWeightPair> outputParents = outputNodes.get(k).parents;

			// each parent of output i.e. each hidden node
			for (int j = 0; j < outputParents.size(); j++) {
				deltaJK[j][k] = learningRate * outputParents.get(j).node.getOutput() * deltaK[k];
//						+ momentumRate * momentumJK[j][k];
//				if (Math.random() > .99999999) {
//					System.out.println(deltaJK[j][k]);
//				}
				// if (deltaJK[j][k] == 0) {
				// jkZero++;
				// }
				// count++;
				// if (deltaJK[j][k] == 0 && Math.random() > .9999999) {
				// System.out.println("JK is zero: " + outputParents.get(j).node.getSum() + " "
				// + outputParents.get(j).node.getOutput() + " " + deltaK[k]);
				// }
				// System.out.print(deltaJK[j][k]);
			}
			// System.out.println();
		}

		// System.out.println("JKZero: " + (double)jkZero/count + "% zero, total: " + count);

		// count = 0;
		// int ijZero = 0;
		for (int j = 0; j < hiddenNodes.size() - 1; j++) {
			if (hiddenNodes.get(j).getSum() < 0) {
				continue;
			}

			double deltaJ = 0;

			// calculate deltaJ
			for (int k = 0; k < weightsJK[j].length; k++) {
				deltaJ += weightsJK[j][k] * deltaK[k];
			}
			ArrayList<NodeWeightPair> hiddenParents = hiddenNodes.get(j).parents;
			// calculate deltaIJ
			for (int i = 0; i < hiddenParents.size(); i++) {
				deltaIJ[i][j] = learningRate * hiddenParents.get(i).node.getOutput() * deltaJ;
//						+ learningRate * regularizationRate * weightsIJ[i][j] + momentumRate * momentumIJ[i][j];
//				if (Math.random() > .99999999) {
//					System.out.println(deltaIJ[i][j]);
//				}
				// if (deltaIJ[i][j] == 0) {
				// ijZero++;
				// }
				// count++;
				// if (deltaIJ[i][j] == 0 && Math.random() > .9999999) {
				// System.out.println("IJ is zero: " + hiddenParents.get(i).node.getSum() + " "
				// + hiddenParents.get(i).node.getOutput() + " " + deltaJ);
				// }
				// System.out.print(deltaIJ[i][j]);
			}
			// System.out.println();
			// System.out.println("IJZero: " + (double)ijZero/count + "% zero, total: " + count);

		}
	}

	/** Apply the changes stored in the deltas. */
	private void applyDeltas() {
		for (int k = 0; k < outputNodes.size(); k++) {
			ArrayList<NodeWeightPair> outputParents = outputNodes.get(k).parents;
			for (int j = 0; j < outputParents.size(); j++) {
				weightsJK[j][k] += deltaJK[j][k] - learningRate * regularizationRate * weightsJK[j][k];
				// if (Math.random() > .999999999)
				// System.out.println("dJK " + deltaJK[j][k] + " + " + weightsJK[j][k]);
				outputParents.get(j).weight = weightsJK[j][k];
			}
		}

		for (int j = 0; j < hiddenNodes.size() - 1; j++) {
			ArrayList<NodeWeightPair> hiddenParents = hiddenNodes.get(j).parents;
			for (int i = 0; i < hiddenParents.size(); i++) {
				weightsIJ[i][j] += deltaIJ[i][j] - learningRate * regularizationRate * weightsIJ[i][j];
				// if (Math.random() > .99999999)
				// System.out.println("dIJ " + deltaIJ[i][j] + " + " + weightsIJ[i][j]);
				hiddenParents.get(i).weight = weightsIJ[i][j];
			}
		}
	}

	/** Copies the contents of src into dst. Assumes arrays have matching size. */
	private static void arrayCopy2D(double[][] src, double[][] dst) {
		for (int i = 0; i < src.length; i++) {
			System.arraycopy(src[i], 0, dst[i], 0, src[i].length);
		}
	}
}
