/**
 * Class to identify connections between different layers.
 */

public class DeepNodeWeightPair
{
	public DeepNode node; // The parent node
	public Double weight; // Weight of this connection

	public DeepNodeWeightPair(DeepNode node, Double weight)
	{
		this.node = node;
		this.weight = weight;
	}
}