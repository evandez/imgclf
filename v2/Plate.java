package v2;

import static v2.Util.checkNotNull;
import static v2.Util.checkPositive;

/** Represents something image-like. */
public class Plate {
	private final double[][][] values;
	
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
	
	/** Returns the result of convolving the given mask with this plate. */
	public Plate convolve(Plate mask) {
		double[][][] result = new double[getNumChannels()][getHeight()][getWidth()];
		// TODO: Implement this method.
		return mask;
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
		double[][][] output = new double[getHeight()][getWidth()][getNumChannels()];
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				for (int k = 0; k < getNumChannels(); k++) {
					output[i][j][k] = func.apply(values[i][j][k]);
				}
			}
		}
		return new Plate(output);
	}
	
	/** Pack this plate into a 1D array, channel by channel, row by row. */
	public double[] as1DArray(Plate plate) {
		// TODO: Implement this method.
		return null;
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
