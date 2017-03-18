package v2;

/**
 * @ Author: Yuting Liu
 * 
 * This is the class for each image instance
 */

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;


public class Instance {
	// store the bufferedImage
    private BufferedImage image;
	private String        label, fileName;
	private int           width, height;
	// separate rgb channels
	private int[][] red_channel, green_channel, blue_channel, gray_image;
	private HowCreated provenance = HowCreated.Original;
	
	protected enum HowCreated {Original, Rotated, Shifted, FlippedTopToBottom, FlippedLeftToRight};
    
	// Constructor
	// given the bufferedimage and its class label
	// get an instance
	public Instance(BufferedImage image, String fileName, String label) {
		this.image    = image;
		this.fileName = fileName;
		this.label    = label;
		width         = image.getWidth();
		height        = image.getHeight();
		
		gray_image = null;
		// get separate rgb channels
		red_channel   = new int[height][width];
		green_channel = new int[height][width];
		blue_channel  = new int[height][width];
		
		for(int row = 0; row < height; ++row) {
			for(int col = 0; col < width; ++col) {
				Color c = new Color(image.getRGB(col, row));
				red_channel[  row][col] = c.getRed();
				green_channel[row][col] = c.getGreen();
				blue_channel[ row][col] = c.getBlue();
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
		// avoid repeated conversion if get the gray image before
		if(gray_image != null) {
			return gray_image;
		}
		
		gray_image = new int[height][width];
		
		// Gray filter
		for(int row = 0; row < height; ++row) {
			for(int col = 0; col < width; ++col) {
				int rgb = image.getRGB(col, row) & 0xFF;
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >>  8) & 0xFF;
				int b = (rgb        & 0xFF);
				gray_image[row][col] = (r + g + b) / 3;
			}
		}
		return gray_image;
	}

	public Instance() { // An empty instance constructor.
	}
	// Take an instance and make a new image with the pixels shifted.  DeltaX and deltaY can be negative.
	public Instance shiftImage(int deltaX, int deltaY) { // The shifted pixels 'wrap around' the other side of the image.
		Instance newInstance = new Instance();	
		newInstance.provenance = HowCreated.Shifted;
		this.getGrayImage(); // Make sure GRAY pixels calculated before copying.
		
		deltaX =  Math.max(-this.height + 1, Math.min(deltaX, this.height - 1)); // Probably should print a warning ...
		deltaY =  Math.max(-this.width  + 1, Math.min(deltaY, this.width  - 1));
		
		newInstance.image  = null;
		newInstance.label  = this.label;
		newInstance.width  = this.width;
		newInstance.height = this.height;
		
		newInstance.red_channel   = new int[height][width];
		newInstance.green_channel = new int[height][width];
		newInstance.blue_channel  = new int[height][width];
		newInstance.gray_image    = new int[height][width];
		
		newInstance.fileName      = this.fileName.replace(".jpg", "_shift" + deltaX + "x" + deltaY + ".jpg");
		
		for(int row = 0; row < this.height; ++row) {
			for(int col = 0; col < this.width; ++col) {
				int newRow = (this.height + row + deltaY) % this.height; // Make sure the argument to MOD is not negative.
				int newCol = (this.width  + col + deltaX) % this.width;
						
				newInstance.red_channel[  newRow][newCol] = this.red_channel[  row][col];
				newInstance.green_channel[newRow][newCol] = this.green_channel[row][col];
				newInstance.blue_channel[ newRow][newCol] = this.blue_channel[ row][col];
				newInstance.gray_image[   newRow][newCol] = this.gray_image[   row][col];
			}
		}
		return newInstance;
	}
	
	// Take an instance and make a new image that is rotated around the "y axis."
	public Instance flipImageLeftToRight() {
		Instance newInstance = new Instance();
		newInstance.provenance = HowCreated.FlippedLeftToRight;
		this.getGrayImage(); // Make sure GRAY pixels calculated before copying.	
		
		newInstance.image  = null;
		newInstance.label  = this.label;
		newInstance.width  = this.width;
		newInstance.height = this.height;
		
		newInstance.red_channel   = new int[height][width];
		newInstance.green_channel = new int[height][width];
		newInstance.blue_channel  = new int[height][width];
		newInstance.gray_image    = new int[height][width];
		
		newInstance.fileName      = this.fileName.replace(".jpg", "_flipLeftToRight.jpg");
		
		for(int row = 0; row < this.height; ++row) {
			for(int col = 0; col < this.width; ++col) {
				int newRow = row;
				int newCol = this.width - col - 1;
						
				newInstance.red_channel[  newRow][newCol] = this.red_channel[  row][col];
				newInstance.green_channel[newRow][newCol] = this.green_channel[row][col];
				newInstance.blue_channel[ newRow][newCol] = this.blue_channel[ row][col];
				newInstance.gray_image[   newRow][newCol] = this.gray_image[   row][col];
			}
		}
		return newInstance;
	}
	
	// Take an instance and make a new image that is rotated around the "x axis."
	public Instance flipImageTopToBottom() {
		Instance newInstance = new Instance();
		newInstance.provenance = HowCreated.FlippedTopToBottom;	
		this.getGrayImage(); // Make sure GRAY pixels calculated before copying.
		
		newInstance.image  = null;
		newInstance.label  = this.label;
		newInstance.width  = this.width;
		newInstance.height = this.height;
		
		newInstance.red_channel   = new int[height][width];
		newInstance.green_channel = new int[height][width];
		newInstance.blue_channel  = new int[height][width];
		newInstance.gray_image    = new int[height][width];
		
		newInstance.fileName      = this.fileName.replace(".jpg", "_flipTopToBottom.jpg");
		
		for(int row = 0; row < this.height; ++row) {
			for(int col = 0; col < this.width; ++col) {
				int newRow = this.height - row - 1;
				int newCol = col;
						
				newInstance.red_channel[  newRow][newCol] = this.red_channel[  row][col];
				newInstance.green_channel[newRow][newCol] = this.green_channel[row][col];
				newInstance.blue_channel[ newRow][newCol] = this.blue_channel[ row][col];
				newInstance.gray_image[   newRow][newCol] = this.gray_image[   row][col];
			}
		}
		return newInstance;
	}
	
	// Rotate in the plane.  Wraps around where needed, hence this will distort a rectanglar image a good del.
	public Instance rotateImageThisManyDegrees(double degrees) {
		Instance newInstance = new Instance();
		newInstance.provenance = HowCreated.Rotated;	
		this.getGrayImage(); // Make sure GRAY pixels calculated before copying.
		
		double radians = Math.toRadians(degrees);
				
		newInstance.image  = null;
		newInstance.label  = this.label;
		newInstance.width  = this.width;
		newInstance.height = this.height;
		
		newInstance.red_channel   = new int[height][width];
		newInstance.green_channel = new int[height][width];
		newInstance.blue_channel  = new int[height][width];
		newInstance.gray_image    = new int[height][width];
		
		newInstance.fileName      = this.fileName.replace(".jpg", "_rotate" + degrees + ".jpg");
		
		for(    int newRow = 0; newRow < this.height; ++newRow) { // See http://stackoverflow.com/questions/2278414/rotating-an-image-in-c-c
			for(int newCol = 0; newCol < this.width;  ++newCol) { // Use the new indices so we make sure something goes into every new cell.  Use -radians since we are rotating BACK to the original image.
				int row = (this.height + (int)(-Math.sin(-radians) * newCol + Math.cos(-radians) * newRow)) % this.height;
				int col = (this.width  + (int)( Math.cos(-radians) * newCol + Math.sin(-radians) * newRow)) % this.width;
						
				newInstance.red_channel[  newRow][newCol] = this.red_channel[  row][col];
				newInstance.green_channel[newRow][newCol] = this.green_channel[row][col];
				newInstance.blue_channel[ newRow][newCol] = this.blue_channel[ row][col];
				newInstance.gray_image[   newRow][newCol] = this.gray_image[   row][col];
			}
		}
		return newInstance;
	}
	
	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
	
	public String getFileName() {
		return fileName;
	}
	
	public String getLabel() {
		return label;
	}
    
    // display the given bitmap
    public void display2D(int[][] img) {
        BufferedImage bufferedImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for(int row = 0; row < height; ++row) {
            for(int col = 0; col < width; ++col) {
                int c = img[row][col] << 16 | img[row][col] << 8 | img[row][col];
                bufferedImg.setRGB(col, row, c);
            }
        }
        displayImage(bufferedImg);
    }
    
    // display the buffered image in the panel  I AM USING Java 1.7 IN THE BMI POOL and 'img' can't be inherited this way
    public void displayImage(BufferedImage img) {
        JFrame frame = new JFrame("Image");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new JPanel() {
			private static final long serialVersionUID = 1L;

			@Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2d = (Graphics2D) g;
                g2d.clearRect(0, 0, getWidth(), getHeight());
                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                                     RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2d.scale(2, 2);
              //  g2d.drawImage(img, 0, 0, this); Fails in Java 1.6 and 1.7
            }
        };
        panel.setPreferredSize(new Dimension(width * 2, height * 2));
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

	public HowCreated getProvenance() {
		return provenance;
	}
}
