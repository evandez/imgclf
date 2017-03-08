package v2;

import java.util.Arrays;

/** Simple sanity checks for problematic methods. */
public class Test {
	public static void main(String[] args) {
		double[][][] data = new double[][][]{
			 {{1.0, 2.0, 3.0},
			  {4.0, 5.0, 6.0},
			  {7.0, 8.0, 9.0}}
		};
		
		// Test plate functions.
		Plate thePlate = new Plate(data);
		System.out.println(thePlate);
		System.out.println(thePlate.rot180());
		System.out.println(thePlate.maxPool(2, 2));

		// Test util functions.
		System.out.println(new Plate(Util.tensorAdd(data, data)));
		System.out.println(new Plate(Util.tensorSubtract(data, data)));
		System.out.println(new Plate(Util.scalarMultiply(3, data)));
		
		// Test convolution. Example taken from below:
		// https://www.researchgate.net/figure/282997080_fig11_Figure-11-An-example-of-matrix-convolution
		Plate img = new Plate(new double[][][]{
			// Channel 1. 
			{{22, 15, 1, 3, 60},
			 {42, 5, 38, 39, 7},
			 {28, 9, 4, 66, 79},
			 {0, 82, 45, 12, 17},
			 {99, 14, 72, 51, 3}},
			 
			 // Channel 2.
			{{22, 15, 1, 3, 60},
			 {42, 5, 38, 39, 7},
			 {28, 9, 4, 66, 79},
			 {0, 82, 45, 12, 17},
			 {99, 14, 72, 51, 3}}
		});
		Plate mask = new Plate(new double[][][]{
			{{0, 0, 1},
			 {0, 0, 0},
			 {0, 0, 0}}
		});
		System.out.println(img.convolve(mask));
		
		// Test flattening.
		Plate pleaseFlattenMe = new Plate(new double[][][]{
			{{22, 15, 1, 3, 60},
			 {42, 5, 38, 39, 7},
			 {28, 9, 4, 66, 79},
			 {0, 82, 45, 12, 17},
			 {99, 14, 72, 51, 3}},
			
			{{1, 2, 3, 4, 5},
			 {6, 7, 8, 9, 10},
			 {11, 12, 13, 14, 15},
			 {16, 17, 18, 19, 20},
			 {21, 22, 23, 24, 25}}
		});
		System.out.println(Arrays.toString(pleaseFlattenMe.as1DArray()));
	}
}
