import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

def embed_watermark(image, watermark, alpha=0.3):
    """
    Embed a watermark into an image using Fourier transform (additive in magnitude spectrum).
    """
    height, width = image.shape
    watermark_height, watermark_width = watermark.shape
    if watermark_height > height or watermark_width > width:
        watermark = cv2.resize(watermark, (width // 4, height // 4))
    watermark_height, watermark_width = watermark.shape

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)

    center_y, center_x = height // 2, width // 2
    top_y = center_y - watermark_height // 2
    left_x = center_x - watermark_width // 2

    # Add watermark to the magnitude spectrum
    for y in range(watermark_height):
        for x in range(watermark_width):
            # Add grayscale watermark intensity to magnitude
            magnitude[top_y + y, left_x + x] += alpha * watermark[y, x]
            # Add symmetric component
            magnitude[height - (top_y + y) - 1, width - (left_x + x) - 1] += alpha * watermark[y, x]

    # Reconstruct the complex DFT
    dft_shift_watermarked = magnitude * np.exp(1j * phase)
    dft_ishift = np.fft.ifftshift(dft_shift_watermarked)
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back

def extract_watermark(original_image, watermarked_image, watermark_size, alpha=0.3):
    """
    Extract the watermark from a watermarked image (magnitude subtraction).
    """
    dft_original = np.fft.fftshift(np.fft.fft2(original_image))
    dft_watermarked = np.fft.fftshift(np.fft.fft2(watermarked_image))
    mag_orig = np.abs(dft_original)
    mag_wm = np.abs(dft_watermarked)
    height, width = original_image.shape
    watermark_height, watermark_width = watermark_size
    center_y, center_x = height // 2, width // 2
    top_y = center_y - watermark_height // 2
    left_x = center_x - watermark_width // 2
    # Subtract the magnitude spectra to recover watermark
    raw_extracted = mag_wm[top_y:top_y+watermark_height, left_x:left_x+watermark_width] - mag_orig[top_y:top_y+watermark_height, left_x:left_x+watermark_width]
    # Recover the watermark intensities
    extracted = np.clip(raw_extracted / alpha, 0, 255).astype(np.uint8)
    return extracted

def visualize_spectrum(image, title="Magnitude Spectrum"):
    """
    Visualize the Fourier transform magnitude spectrum of an image.
    
    Args:
        image (numpy.ndarray): Input image
        title (str): Plot title
    """
    # Compute the 2D Fourier Transform
    f = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    fshift = np.fft.fftshift(f)
    
    # Calculate the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Display the results
    plt.figure(figsize=(10, 7))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def example_usage():
    """
    Example usage of the watermarking functions.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    try:
        # Replace with your actual image paths
        original_image = load_image("input/ghibli_image.png")
        watermark = load_image("input/openai_watermark.png")
        
        # Embed watermark
        watermarked_image = embed_watermark(original_image, watermark, alpha=0.3)
        
        # Save watermarked image
        cv2.imwrite(os.path.join(output_dir, "watermarked.png"), watermarked_image)
        
        # Extract watermark
        extracted_watermark = extract_watermark(original_image, watermarked_image, watermark.shape, alpha=0.3)
        
        # Save extracted watermark
        cv2.imwrite(os.path.join(output_dir, "extracted_watermark.png"), extracted_watermark)
        
        # Visualize
        visualize_spectrum(original_image, "Original Image Spectrum")
        visualize_spectrum(watermarked_image, "Watermarked Image Spectrum")
        
        # Display the results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(221), plt.imshow(original_image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(222), plt.imshow(watermark, cmap='gray')
        plt.title('Watermark'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(223), plt.imshow(watermarked_image, cmap='gray')
        plt.title('Watermarked Image'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(224), plt.imshow(extracted_watermark, cmap='gray')
        plt.title('Extracted Watermark'), plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "results.png"))
        plt.show()
        
        print("Watermarking process completed successfully!")
        print(f"All output files saved to {output_dir}/ directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have valid input images in the 'input' directory.")
        print("Required files: 'input/original.png' and 'input/watermark.png'")

if __name__ == "__main__":
    # Uncomment to run the example:
    example_usage()
    
    print("Fourier Transform Watermarking Tool")
    print("==================================")
    print("1. Create input/ and output/ directories")
    print("2. Place your original image as 'input/original.png'")
    print("3. Place your watermark as 'input/watermark.png'")
    print("4. Uncomment the example_usage() call in this script")
    print("5. Run the script to generate watermarked images and results")
