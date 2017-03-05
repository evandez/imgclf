package v2;

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
		System.out.println(thePlate.toString());
		System.out.println(thePlate.rot180().toString());
		System.out.println(thePlate.maxPool(2, 2).toString());

		// Test util functions.
		System.out.println(new Plate(Util.tensorAdd(data, data)).toString());
		System.out.println(new Plate(Util.tensorSubtract(data, data)).toString());
		System.out.println(new Plate(Util.scalarMultiply(3, data)).toString());
		
		// Test convolution. Example taken from below:
		// https://www.researchgate.net/figure/282997080_fig11_Figure-11-An-example-of-matrix-convolution
		Plate img = new Plate(new double[][][]{
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
		System.out.println(img.convolve(mask).toString());
	}
}
