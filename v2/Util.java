package v2;

import java.util.Collection;
import java.util.Random;

/** Utility methods and objects used throughout the network. */
public final class Util {
	public static final int SEED = 0;
	public static final Random RNG = new Random(SEED);
	
	/** Performs tensor (3D matrix) scalar multiplication. */
	public static double[][][] scalarMultiply(double scalar, double[][][] tensor) {
		checkTensorNotNullOrEmpty(tensor);
		double[][][] result = new double[tensor.length][tensor[0].length][tensor[0][0].length];
		for (int i = 0; i < tensor.length; i++) {
			for (int j = 0; j < tensor[i].length; j++) {
				for (int k = 0; k < tensor[i][j].length; k++) {
					result[i][j][k] = tensor[i][j][k] * scalar;
				}
			}
 		}
		return result;
	}
	
	/** Performs t1 + t2. */
	public static double[][][] tensorAdd(double[][][] t1, double[][][] t2) {
		checkTensorNotNullOrEmpty(t1);
		checkTensorNotNullOrEmpty(t2);
		checkTensorDimensionsMatch(t1, t2);
		double[][][] result = new double[t1.length][t1[0].length][t1[0][0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				for (int k = 0; k < result[i][j].length; k++) {
					result[i][j][k] = t1[i][j][k] + t2[i][j][k];
				}
			}
		}
		return result;
	}
	
	/** Performs t1 - t2. */
	public static double[][][] tensorSubtract(double[][][] t1, double[][][] t2) {
		return tensorAdd(t1, scalarMultiply(-1, t2));
	}

	/** Verifies that the tensor is not null and that all 3 dimensions have length > 0. */
	public static void checkTensorNotNullOrEmpty(double[][][] tensor) {
		checkNotNull(tensor, "Tensor arg");
		checkPositive(tensor.length, "Tensor dimension 1", false);
		checkPositive(tensor[0].length, "Tensor dimension 2", false);
		checkPositive(tensor[0][0].length, "Tensor dimension 3", false);
	}
	
	/** Verifies that the tensors have the same dimensions. */
	public static void checkTensorDimensionsMatch(double[][][] t1, double[][][] t2) {
		if (t1.length != t2.length 
				|| t1[0].length != t2[0].length
				|| t1[0][0].length != t2[0][0].length) {
			throw new IllegalArgumentException(
					String.format(
							"Tensor dimensions do not match...\tT1:%dx%dx%d\tT2:%dx%dx%d\n",
							t1.length,
							t1[0].length,
							t1[0][0].length,
							t2.length,
							t2[0].length,
							t2[0][0].length));
		}
	}
	
	/**
	 * Verifies that the value with the given name is in the range [min, max).
	 */
	public static void checkValueInRange(int val, int min, int max, String name) {
		if (val < min || val >= max) {
			throw new IllegalArgumentException(
					String.format(
							"%s was %d, but should be in range [%d, %d)", name, val, min, max));
		}
	}
	
	/** Verifies that the object with the given name is not null. */
	public static void checkNotNull(Object obj, String name) {
		if (obj == null) {
			throw new NullPointerException(String.format("%s was null!", name));
		}
	}
	
	/**
	 * Verifies that the value with the given name is not null.
	 * 
	 * The boolean parameter specifies which type of exception to throw.
	 */
	public static void checkPositive(double val, String name, boolean sourceIsStateful) {
		if (val <= 0) {
			if (sourceIsStateful) {
				throw new IllegalStateException(String.format("%s was not set!", name));
			} else {				
				throw new IllegalArgumentException(String.format("%s must be positive!", name));
			}
		}
	}
	
	/**
	 * Checks that the collection with the given name is nonempty.
	 * 
	 * Again, the boolean parameter specifies the type of exception to throw.
	 */
	public static void checkNotEmpty(Collection<?> coll, String name, boolean sourceIsStateful) {
		if (coll.isEmpty()) {
			if (sourceIsStateful) {
				throw new IllegalStateException(
						String.format("%s must have at least one value!", name));
			} else {
				throw new IllegalArgumentException(String.format("%s must be nonempty!", name));
			}
		}
	}
}
