package v2;

import java.util.ArrayList;
import java.util.List;

/**
 * @Author: Yuting Liu
 * This is the dataset class that holds in the whole dataset
 *
 */

public class Dataset {
	// the list of all instances
	private ArrayList<Instance> instances;

	Dataset() {
		this.instances = new ArrayList<>();
	}

	// get the size of the dataset
	public int getSize() {
		return instances.size();
	}

	// add instance into the data set
	public void add(Instance inst) {
		instances.add(inst);
	}

	// Return the list of images.
	List<Instance> getImages() {
		return instances;
	}
}
