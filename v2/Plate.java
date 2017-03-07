package v2;

import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;
import static v2.Util.checkValueInRange;

/** Represents something image-like. */
public class Plate {
	private final double[][][] values;
	
	/**
	 * Constructs a new plate for the given values. 
	 * 
	 * IMPORTANT: The values should be organized so that the dimensions follow the
	 * pattern of (channels, height, width).
	 */
	public Plate(double[][][] values) {
		checkNotNull(values, "Plate values");
		checkPositive(values.length, "Plate channels", false);
		checkPositive(values[0].length, "Plate height", false);
		checkPositive(values[0][0].length, "Plate width", false);
		this.values = values;
	}

	/** Returns the number of channels in the plate. */
	public int getNumChannels() { return values.length; }
	
	/** Returns the height of each channel. */
	public int getHeight() { return values[0].length; }
	
	/** Returns the width of each channel. */
	public int getWidth() { return values[0][0].length; }
	
	/** Returns the total number of values in the plate. */
	public int getTotalNumValues() { return getNumChannels() * getHeight() * getWidth(); }
	
	/** Returns the value at the given channel, row, and column. */
	public double valueAt(int chan, int row, int col) {
		checkValueInRange(chan, 0, getNumChannels(), "Channel index");
		checkValueInRange(row, 0, getHeight(), "Row index");
		checkValueInRange(col, 0, getWidth(), "Column index");
		return values[chan][row][col];
	}

	/**
	 * Returns the result of convolving the given mask with this plate.
	 * 
	 * The returned plate has the same size as this one.
	 */
	public Plate convolve(Plate mask) {
		if (getNumChannels() != mask.getNumChannels()) {
			throw new IllegalArgumentException("Mask must have same number of channels as plate.");
		} else if (getHeight() < mask.getHeight() || getWidth() < mask.getWidth()) {
			throw new IllegalArgumentException("Mask must be smaller than plate.");
		}
		double[][][] result = new double[getNumChannels()][getHeight()][getWidth()];
		for (int chan = 0; chan < getNumChannels(); chan++) {
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++) {
					result[chan][i][j] = convolvePixelIJ(mask, chan, i, j);
				}
			}
		}
		return new Plate(result);
	}
	
	private double convolvePixelIJ(Plate mask, int chan, int i, int j) {
		double sum = 0.0;
		for (int k = 0; k < mask.getHeight(); k++) {
			for (int l = 0; l < mask.getWidth(); l++) {
				int neighborX = i - (mask.getHeight() / 2) + k;
				int neighborY = j - (mask.getWidth() / 2) + l;
				if (neighborX < 0 || neighborX >= getHeight() 
						|| neighborY < 0 || neighborY >= getWidth()) {
					continue;
				}
				sum += mask.values[chan][k][l] * values[chan][neighborX][neighborY];
			}
		}
		return sum;
	}
	
	/** Flips each channel by 180 degrees. */
	public Plate rot180() {
		double[][][] result = new double[getNumChannels()][getHeight()][getWidth()];
		for (int chan = 0; chan < getNumChannels(); chan++) {
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++) {
					result[chan][i][j] = values[chan][getHeight()-1-i][getWidth()-1-j];
				}
			}
		}
		return new Plate(result);
	}
	
	/** Returns the max-pooled plate. No overlap between each pool. */
	public Plate maxPool(int windowHeight, int windowWidth) {
		checkValueInRange(windowHeight, 0, getHeight(), "Max pool window height");
		checkValueInRange(windowWidth, 0, getWidth(), "Max pool window width");
		int resultHeight = getHeight() / windowHeight;
		int resultWidth = getWidth() / windowWidth;
		resultHeight += getHeight() % windowHeight == 0 ? 0 : 1;
		resultWidth += getWidth() % windowWidth == 0 ? 0 : 1;
		
		double[][][] result = new double[getNumChannels()][resultHeight][resultWidth];
		for (int chan = 0; chan < getNumChannels(); chan++) {
			for (int i = 0; i < result[chan].length; i++) {
				for (int j = 0; j < result[chan][i].length; j++) {
					int windowStartI = Math.min(i * windowHeight, getHeight() - 1);
					int windowStartJ = Math.min(j * windowWidth, getWidth() - 1);
					result[chan][i][j] =
							maxValInWindow(
									chan,
									windowStartI,
									windowStartJ,
									windowHeight,
									windowWidth);
				}
 			}
		}
		return new Plate(result);
	}
	
	private double maxValInWindow(
			int chan, int windowStartI, int windowStartJ, int windowHeight, int windowWidth) {
		double max = Double.MIN_VALUE;
		int windowEndI = Math.min(windowStartI + windowHeight - 1, getHeight() - 1);
		int windowEndJ = Math.min(windowStartJ + windowWidth - 1, getWidth() - 1);
		for (int i = windowStartI; i <= windowEndI; i++) {
			for (int j = windowStartJ; j <= windowEndJ; j++) {
				max = Math.max(max, values[chan][i][j]);
			}
		}
		return max;
	}
	
	/** Applies the activation function to all values in the plate. */
	public Plate applyActivation(ActivationFunction func) {
		checkNotNull(func, "Activation function");
		double[][][] output = new double[getNumChannels()][getHeight()][getWidth()];
		for (int chan = 0; chan < getNumChannels(); chan++) {
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++) {
					output[chan][i][j] = func.apply(values[chan][i][j]);
				}
			}
		}
		return new Plate(output);
	}
	
	/** Pack this plate into a 1D array, channel by channel, row by row. */
	public double[] as1DArray() {
		double[] result = new double[getTotalNumValues()];
		for (int chan = 0; chan < getNumChannels(); chan++) {
			for (int row = 0; row < getHeight(); row++) {
				System.arraycopy(
						values[chan][row],
						0 /* Copy the whole row! */,
						result,
						chan * getHeight() * getWidth() + row * getWidth(),
						getWidth());
			}
		}
		return result;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append(
				String.format(
						"Plate with dimensions %dx%dx%d:\n",
						getHeight(),
						getWidth(),
						getNumChannels()));
		for (int chan = 0; chan < getNumChannels(); chan++) {
			builder.append(String.format("chan %d = [\n", chan));
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++) {
					builder.append(String.format("%f, ", values[chan][i][j]));
				}
				builder.append("\n");
			}
			builder.append("]\n");
		}
		return builder.toString();
	}
}
