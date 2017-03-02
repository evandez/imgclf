
///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  NNBuilder.java
// File:             Node.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.ArrayList;

/**
 * Class for internal organization of a Neural Network. There are 5 types of nodes.
 */
public class Node {
	// 0 = input, 1 = biasToHidden, 2 = hidden, 3 = biasToOutput, 4 = Output
	private int type = 0;
	// Array List that will contain the parents (including the bias node) with weights if applicable
	public ArrayList<NodeWeightPair> parents = null;

	Double inputValue = 0.0;
	Double outputValue = 0.0;
	private Double sum = 0.0; // sum of wi*xi

	// Create a node with a specific type
	public Node(int type) {
		if (type > 4 || type < 0) {
			throw new RuntimeException("Invalid node type!");
		} else {
			this.type = type;
		}

		if (this.type == 2 || this.type == 4) {
			parents = new ArrayList<NodeWeightPair>();
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
	 * Calculate the output of a Sigmoid node.
	 * 
	 * @param train:
	 *           the training set
	 */
	public double computeOutput() {
		if (type == 2 || type == 4) {
			sum = 0.0;
			// boolean pos = true;
			for (int i = 0; i < parents.size(); i++) {
				// if (sum > 0) {
				// pos = true;
				// } else if (sum < 0){
				// System.out
				// .println("neg: " + parents.get(i).weight + " " + parents.get(i).node.getOutput() + " sum: " + sum);
				// pos = false;
				// }
				sum += parents.get(i).weight * parents.get(i).node.getOutput();
				// if (pos == true && sum < 0 && parents.get(i).weight > 0 && parents.get(i).node.getOutput() > 0) {
				// System.out
				// .println("OVERFLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW " + sum);
				// }
			}
			// if (type == 4) {
//			if (Math.random() > .99) {
//				System.out.println("sum: " + sum);
//			}
//			outputValue = sum = 1 / (1 + Math.exp(-sum));
			// } else {
			 outputValue = sum = Math.max(0.0, sum);
			// }
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
