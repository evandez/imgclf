package cnn.components;

import static cnn.tools.Util.checkNotNull;
import static cnn.tools.Util.checkPositive;
import static cnn.tools.Util.checkValueInRange;

import cnn.tools.ActivationFunction;

/** Represents something image-like. */
public class Plate {
	private double[][] values;
	
	/**
	 * Constructs a new plate for the given values. 
	 * 
	 * IMPORTANT: The values should be organized so that the dimensions follow the
	 * pattern of (height, width).
	 */
	public Plate(double[][] values) {
		checkNotNull(values, "Plate values");
		checkPositive(values.length, "Plate height", false);
		checkPositive(values[0].length, "Plate width", false);
		this.values = values;
	}

	/** Returns the height of each channel. */
	public int getHeight() { return values.length; }
	
	/** Returns the width of each channel. */
	public int getWidth() { return values[0].length; }
	
	/** Returns the total number of values in the plate. */
	public int getTotalNumValues() { return getHeight() * getWidth(); }
	
	public double[][] getValues() {
		return values;
	}
	
	public void setVals(double[][] newVals) {
		this.values = newVals;
	}
	
	/** Returns the value at the given channel, row, and column. */
	public double valueAt( int row, int col) {
		checkValueInRange(row, 0, getHeight(), "Row index");
		checkValueInRange(col, 0, getWidth(), "Column index");
		return values[row][col];
	}

	/**
	 * Returns the result of convolving the given mask with this plate.
	 * 
	 * The returned plate has the same size as this one.
	 */
	public Plate convolve(Plate mask) {
		checkValidMask(mask);
		int maskHeight = mask.getHeight();
		int maskWidth = mask.getWidth();
		double[][] result = new double[getHeight() - maskHeight + 1][getWidth() - maskWidth + 1];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = convolvePixelIJ(mask, i + maskHeight / 2, j + maskWidth / 2);
			}
		}
		return new Plate(result);
	}
	
	private double convolvePixelIJ(Plate mask, int i, int j) {
		double sum = 0.0;
		for (int k = 0; k < mask.getHeight(); k++) {
			for (int l = 0; l < mask.getWidth(); l++) {
				int neighborX = i - (mask.getHeight() / 2) + k;
				int neighborY = j - (mask.getWidth() / 2) + l;
				if (neighborX < 0 || neighborX >= getHeight() 
						|| neighborY < 0 || neighborY >= getWidth()) {
					continue;
				}
				sum += mask.values[k][l] * values[neighborX][neighborY];
			}
		}
		return sum;
	}
	
	private void checkValidMask(Plate mask) {
		if (getHeight() < mask.getHeight() || getWidth() < mask.getWidth()) {
			throw new IllegalArgumentException("Mask must be smaller than plate.");
		}
	}
	
	/** Flips each channel by 180 degrees. */
	public Plate rot180() {
		double[][] result = new double[getHeight()][getWidth()];
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				result[i][j] = values[getHeight()-1-i][getWidth()-1-j];
			}
		}
		return new Plate(result);
	}
	
	/** Applies the activation function to all values in the plate. */
	public Plate applyActivation(ActivationFunction func) {
		checkNotNull(func, "Activation function");
		double[][] output = new double[getHeight()][getWidth()];
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				output[i][j] = func.apply(values[i][j]);
			}
		}
		return new Plate(output);
	}
	
	/** Pack this plate into a 1D array, channel by channel, row by row. */
	public double[] as1DArray() {
		double[] result = new double[getTotalNumValues()];
		for (int row = 0; row < getHeight(); row++) {
			System.arraycopy(
					values[row],
					0 /* Copy the whole row! */,
					result,
					row * getWidth(),
					getWidth());
		}
		return result;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append(
				String.format(
						"Plate with dimensions %dx%d:\n",
						getHeight(),
						getWidth()));
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				builder.append(String.format("%f, ", values[i][j]));
			}
			builder.append("\n");
		}
		builder.append("]\n");
		return builder.toString();
	}
}
