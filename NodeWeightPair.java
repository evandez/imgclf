///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  NNBuilder.java
// File:             NodeWeightPair.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

/**
 * Class to identify connections between different layers.
 */

public class NodeWeightPair
{
	public Node node; // The parent node
	public Double weight; // Weight of this connection

	public NodeWeightPair(Node node, Double weight)
	{
		this.node = node;
		this.weight = weight;
	}
}