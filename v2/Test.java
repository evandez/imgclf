package v2;

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
	}
}
