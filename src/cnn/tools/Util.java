package cnn.tools;

import java.util.Collection;
import java.util.Random;

/** Utility methods and objects used throughout the network. */
public final class Util {
	public static final int SEED = 0;
	public static final Random RNG = new Random(SEED);
	
	/** Performs v1 * v2^T. */
	public static double[][] outerProduct(double[] v1, double[] v2) {
		checkVectorNotNullOrEmpty(v1);
		checkVectorNotNullOrEmpty(v2);
		double[][] result = new double[v1.length][v2.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = v1[i] * v2[j];
			}
		}
		return result;
	}
	
	/** Computes the inner product of two vectors. */
	public static double innerProduct(double[] v1, double[] v2) {
		checkVectorNotNullOrEmpty(v1);
		checkVectorNotNullOrEmpty(v2);
		checkVectorDimensionsMatch(v1, v2);
		double result = 0;
		for (int i = 0; i < v1.length; i++) {
			result += v1[i] * v2[i];
		}
		return result;
	}
	
	/** Performs vector scalar multiplication. See description for 3D version. */
	public static double[] scalarMultiply(double scalar, double[] vector, boolean inline) {
		return scalarMultiply(scalar, new double[][][]{{ vector }}, inline)[0][0];
	}
	
	/** Performs matrix scalar multiplication. See description for 3D version. */
	public static double[][] scalarMultiply(double scalar, double[][] matrix, boolean inline) {
		return scalarMultiply(scalar, new double[][][]{ matrix }, inline)[0];
	}
	
	/**
	 * Performs tensor (3D matrix) scalar multiplication.
	 * 
	 * If inline is true, this method directly mutates the given tensor.
	 */
	public static double[][][] scalarMultiply(double scalar, double[][][] tensor, boolean inline) {
		checkTensorNotNullOrEmpty(tensor);
		double[][][] result = inline
				? tensor
				: new double[tensor.length][tensor[0].length][tensor[0][0].length];
		for (int i = 0; i < tensor.length; i++) {
			for (int j = 0; j < tensor[i].length; j++) {
				for (int k = 0; k < tensor[i][j].length; k++) {
					result[i][j][k] = tensor[i][j][k] * scalar;
				}
			}
 		}
		return result;
	}
	
	/** Performs v1 - v2 (for vectors). See description for 3D version. */
	public static double[] tensorSubtract(double[] v1, double[] v2, boolean inline) {
		return tensorSubtract(new double[][][] {{ v1 }}, new double[][][]{{ v2 }}, inline)[0][0];
	}
	
	/** Performs m1 - m2 (for matrices). See description for 3D version. */
	public static double[][] tensorSubtract(double[][] m1, double[][] m2, boolean inline) {
		return tensorSubtract(new double[][][]{ m1 }, new double[][][]{ m2 }, inline)[0];
	}
	
	/** Performs t1 - t2 (for 3D tensors). See description for 3D tensorAdd. */
	public static double[][][] tensorSubtract(double[][][] t1, double[][][] t2, boolean inline) {
		return tensorAdd(t1, scalarMultiply(-1, t2, inline), inline);
	}
	
	/** Performs v1 + v2 (for vectors). See description for 3D version. */
	public static double[] tensorAdd(double[] v1, double[] v2, boolean inline) {
		return tensorAdd(new double[][][]{{ v1 }}, new double[][][]{{ v2 }}, inline)[0][0];
	}
	
	/** Performs m1 + m2 (for matrices). See description for 3D version. */
	public static double[][] tensorAdd(double[][] m1, double[][] m2, boolean inline) {
		return tensorAdd(new double[][][]{ m1 }, new double[][][]{ m2 }, inline)[0];
	}
	
	/**
	 * Performs t1 + t2 (for 3D tensors).
	 * 
	 * If inline is true, this method directly mutates t1.
	 */
	public static double[][][] tensorAdd(double[][][] t1, double[][][] t2, boolean inline) {
		checkTensorNotNullOrEmpty(t1);
		checkTensorNotNullOrEmpty(t2);
		checkTensorDimensionsMatch(t1, t2);
		double[][][] result = inline ? t1 : new double[t1.length][t1[0].length][t1[0][0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				for (int k = 0; k < result[i][j].length; k++) {
					result[i][j][k] = t1[i][j][k] + t2[i][j][k];
				}
			}
		}
		return result;
	}
	
	private static void checkVectorNotNullOrEmpty(double[] vector) {
		checkNotNull(vector, "Vector arg");
		checkPositive(vector.length, "Vector length", false);
	}
	
	private static void checkVectorDimensionsMatch(double[] v1, double[] v2) {
		if (v1.length != v2.length) {
			throw new IllegalArgumentException(
					String.format(
							"Lengths %d and %d do not match for inner product.\n",
							v1.length,
							v2.length));
		}
	}
	
	/** Verifies that the tensor is not null and that all 3 dimensions have length > 0. */
	private static void checkTensorNotNullOrEmpty(double[][][] tensor) {
		checkNotNull(tensor, "Tensor arg");
		checkPositive(tensor.length, "Tensor dimension 1", false);
		checkPositive(tensor[0].length, "Tensor dimension 2", false);
		checkPositive(tensor[0][0].length, "Tensor dimension 3", false);
	}
	
	/** Verifies that the tensors have the same dimensions. */
	private static void checkTensorDimensionsMatch(double[][][] t1, double[][][] t2) {
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
	
	/** Verifies that the value with the given name is in the range [min, max). */
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
	
	/** Prints current heap usage. */
	public static void printMemory() {
		// Get current size of heap in bytes
		long heapSize = Runtime.getRuntime().totalMemory(); 

		// Get amount of free memory within the heap in bytes. This size will increase
		// after garbage collection and decrease as new objects are created.
		long heapFreeSize = Runtime.getRuntime().freeMemory(); 

		// Get maximum size of heap in bytes. The heap cannot grow beyond this size.
		// Any attempt will result in an OutOfMemoryException.
		long heapMaxSize = Runtime.getRuntime().maxMemory();
		
		System.out.println("\nCurrent Heap Size: " + heapSize / Math.pow(2,20) + " MB");
		System.out.println("Free Heap Size: " + heapFreeSize / Math.pow(2, 20) + " MB");
		System.out.println("Max Heap Size: " + heapMaxSize / Math.pow(2, 20) + " MB");
	}

}
