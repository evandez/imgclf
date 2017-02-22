/**
 * 
 * This is the class for each image instance
 */

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.image.DataBufferByte;

public class Instance {
	// store the bufferedImage
    private BufferedImage image;
	private String label;
	private int width, height;
	// separate rgb channels
	private int[][] red_channel, green_channel, blue_channel, gray_image;
    
	// Constructor
	// given the bufferedimage and its class label
	// get the 
	public Instance(BufferedImage image, String label) {
		this.image = image;
		this.label = label;
		width = image.getWidth();
		height = image.getHeight();
		
		// get separate rgb channels
		red_channel = new int[height][width];
		green_channel = new int[height][width];
		blue_channel = new int[height][width];
		gray_image = new int[height][width];
		
		for(int row = 0; row < height; ++row) {
			for(int col = 0; col < width; ++col) {
				Color c = new Color(image.getRGB(col, row));
				red_channel[row][col] = c.getRed();
				green_channel[row][col] = c.getGreen();
				blue_channel[row][col] = c.getBlue();
			}
		}
	}
	
	// get separate red channel image
	public int[][] getRedChannel() {
		return red_channel;
	}
	
	// get separate green channel image
	public int[][] getGreenChannel() {
		return green_channel;
	}
	
	// get separate blue channel image
	public int[][] getBlueChannel() {
		return blue_channel;
	}

	// get the gray scale image
	public int[][] getGrayImage() {
		// Gray filter
		BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		byte[] dstBuff = ((DataBufferByte) grayImage.getRaster().getDataBuffer()).getData();
		
		for(int row = 0; row < height; ++row) {
			for(int col = 0; col < width; ++col) {
				gray_image[row][col] = dstBuff[col + row * width] & 0xFF;
			}
		}
		return gray_image;
	}
	
	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
	
	public String getLabel() {
		return label;
	}
}
