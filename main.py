import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter, ArtistAnimation
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

def load_video(image_path):
    """
    Function to load in videos as 3D Numpy arrays
    """
    frames = []

    path = image_path
    cap = cv2.VideoCapture(path)
    ret = True

    while ret:
        ret, img = cap.read()  # read one frame
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            frames.append(img)

    cap.release()

    video = np.stack(frames, axis=0)  # (T, H, W)
    
    return video

def embed_watermark(image, watermark, alpha=0.3):
    """
    Embed a watermark in a watermarked image using magnitude addition
    
    Args:
        image (numpy.ndarray): Image to be watermarked
        watermark (numpy.ndarray): logo/watermark to be embedded in the fourier domain
        optional: alpha: strength of watermark to be applied (stronger = more distortion of watermarked image)
    """

    height, width = image.shape
    watermark = cv2.resize(watermark, (width//2, height//2))

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)

    construct_watermark = np.zeros_like(image)
    construct_watermark[0:watermark.shape[0], 0:watermark.shape[1]] = watermark


    # Add watermark to the central region (using slice assignment instead of loops)
    magnitude += alpha * construct_watermark
    
    # Reconstruct the complex DFT
    dft_shift_watermarked = magnitude * np.exp(1j * phase)
    dft_ishift = np.fft.ifftshift(dft_shift_watermarked)
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back

def embed_video_watermark_TD(video, watermark, alpha=0.3):
    """
    Embed a watermark in a video by marking one pixel on each frame
    
    Args:
        video (numpy.ndarray): video to be watermarked
        watermark (numpy.ndarray): a 1D signature/signal to be applied
        optional: alpha: strength of watermark to be applied
    """

    key1= int(np.random.uniform(low=0, high=video.shape[1]))
    key2 = int(np.random.uniform(low=0, high=video.shape[2]))

    print(key1, key2)

    video_back = video.copy()

    vector = video_back[:, key1, key2]

    vector += (watermark*alpha).astype(np.uint8)

    video_back[:, key1, key2] = np.clip(vector, 0, 255).astype(np.uint8)

    return video_back, key1, key2

def extract_watermark(original_image, watermarked_image, alpha=0.3):
    """
    Extract the watermark from a watermarked image using magnitude subtraction
    
    Args:
        original_image (numpy.ndarray): Original input image
        watermarked_image (numpy.ndarray): Watermarked image for watermark extraction
        optional: alpha: expected strength of watermark
    """
    dft_original = np.fft.fftshift(np.fft.fft2(original_image))
    dft_watermarked = np.fft.fftshift(np.fft.fft2(watermarked_image))
    mag_orig = np.abs(dft_original)
    mag_wm = np.abs(dft_watermarked)

    # Subtract the magnitude spectra to recover watermark
    raw_extracted = mag_wm - mag_orig
    
    raw_extracted = raw_extracted[0:original_image.shape[0]//2, 0: original_image.shape[1]//2]

    raw_extracted = cv2.resize(raw_extracted, (max(raw_extracted.shape), max(raw_extracted.shape)))
    
    # Recover the watermark intensities
    extracted = np.clip(raw_extracted / alpha, 0, 255).astype(np.uint8)
    return extracted

def extract_video_watermark_TD(video, watermarked_video, key1, key2, alpha):
    """
    Extract (Fourier-embedded) watermark from a video 

    Args:
        video (numpy.ndarray): original video
        watermarked_video (numpy.ndarray): the video that has been watermarked
        key1, key2: the pixel location used to embed the watermark
        alpha (float): embedding strength
    """
    # Get the pixel signal across frames
    original_vector = video[:, key1, key2].copy()
    watermarked_vector = watermarked_video[:, key1, key2].copy()

    diff = original_vector.astype(np.int16) - watermarked_vector.astype(np.int16)

    # Extract magnitude difference (watermark = diff / alpha)
    extracted = np.abs((np.fft.fft((diff)/alpha)))

    # Optional: interpolate
    extracted = np.interp((extracted), [np.min(extracted), np.max(extracted)], [0,255]).astype(np.uint8)
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

def example_usage_image():
    """
    Example usage of the watermarking functions.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    try:
        # Replace with your actual image paths
        original_image = load_image("input\ghibli_image.png")
        watermark = load_image("input\openai_watermark.png")
        
        # Embed watermark
        watermarked_image = embed_watermark(original_image, watermark, alpha=0.3)
        
        # Save watermarked image
        cv2.imwrite(os.path.join(output_dir, "watermarked.png"), watermarked_image)
        
        # Extract watermark
        extracted_watermark = extract_watermark(original_image, watermarked_image, alpha=0.3)
        
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
        print("Required files: 'input\original.png' and 'input\watermark.png'")

def example_usage_video():
    """
    Example usage of the watermarking function on each frame of a video
    """

    video = load_video("input\meatthezoo.mp4")
    watermark = cv2.resize(load_image("input\youtube_watermark.jpg"), (video.shape[2], video.shape[1]))
    _, watermark = cv2.threshold(watermark, 100, 255, cv2.THRESH_BINARY)
    plt.imshow(watermark, cmap=cm.gray, vmin=0, vmax=255)
    # plt.show()
    watermarked_video=np.stack([embed_watermark(video[i],watermark,alpha=0.3) for i in range(video.shape[0])])

    visualize_spectrum(video[0], "Original Image Spectrum")
    visualize_spectrum(watermarked_video[0], "Watermarked Image Spectrum")

    fig = plt.figure()
    frames =[]
    for i in range(video.shape[0]):
        frames.append([plt.imshow(watermarked_video[i], cmap=cm.gray,animated=True, vmin=0, vmax=255)])

    ani = ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.axis(False)
    plt.tight_layout()
    # plt.show()
    fig = plt.figure()
    frames =[]

    for i in range(video.shape[0]):
        frames.append([plt.imshow(extract_watermark(video[i], watermarked_video[i], alpha=0.3), cmap=cm.gray,animated=True, vmin=0, vmax=255)])

    ani = ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.axis(False)
    plt.tight_layout()
    plt.show()

def example_usage_video_TD():
    video = load_video("input\meatthezoo.mp4")
    time = np.linspace(0, video.shape[0], video.shape[0])
    freq = 10
    N = len(time)
    watermark = np.sin(2 * np.pi * time * freq / N)

    plt.title("watermark frequency spectrum")
    plt.plot(time, (np.fft.fft(watermark)))
    plt.show()

    watermarked_video, key1, key2 = embed_video_watermark_TD(video, watermark, alpha = 10)

    plt.title("watermarked video vector")
    plt.plot(time, watermarked_video[:, key1, key2])
    plt.show()
    plt.title("video vector")
    plt.plot(time, video[:, key1, key2])
    plt.show()
    plt.title("difference of watermarked and original videos")
    diff = watermarked_video[:, key1, key2].astype(np.int16) - video[:, key1, key2].astype(np.int16)
    plt.plot(time, diff)
    plt.show()

    fig = plt.figure()
    frames =[]
    for i in range(video.shape[0]):
        frames.append([plt.imshow(watermarked_video[i], cmap=cm.gray,animated=True, vmin=0, vmax=255)])

    ani = ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.axis(False)
    plt.tight_layout()
    plt.show()

    extracted_watermark = extract_video_watermark_TD(video, watermarked_video, key1, key2, alpha=10)

    plt.title("Frequency spikes of extracted watermark")
    plt.plot(time, extracted_watermark)

    # Find the index of the max spike in the specified range
    search_range = extracted_watermark[0:len(time)//2]
    peak_index = np.argmax(search_range)
    peak_time = time[peak_index]

    # Add a dashed vertical line at the peak
    plt.axvline(x=peak_time, color='r', linestyle='--', label=f'Peak at {peak_time:.2f}')
    plt.legend()
    plt.show()
    plt.plot(extract_video_watermark_TD(video, watermarked_video, key1-1, key2-1, alpha = 0.1))




if __name__ == "__main__":
    # Uncomment to run the example:

    # example_usage_image()

    # example_usage_video()

    example_usage_video_TD()
    
    # print("Fourier Transform Watermarking Tool")
    # print("==================================")
    # print("1. Create input/ and output/ directories")
    # print("2. Place your original image as 'input/original.png'")
    # print("3. Place your watermark as 'input/watermark.png'")
    # print("4. Uncomment the example_usage() call in this script")
    # print("5. Run the script to generate watermarked images and results")
