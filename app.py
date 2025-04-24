import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def extract_subset(image, x, y, size=20):
    h, w = image.shape
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return image[y:y+size, x:x+size]

def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        row, col = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, (row, col))
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = 0.04
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        return noisy_image
    return image
def apply_frequency_filter(image, filter_type='ideal_low', cutoff=10, order=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2 
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    u = u - crow
    v = v - ccol
    d = np.sqrt(u**2 + v**2)
    h = np.zeros((rows, cols), dtype=np.float32)
    if filter_type == 'ideal_low':
        h[d <= cutoff] = 1
    elif filter_type == 'ideal_high':
        h[d > cutoff] = 1
    elif filter_type == 'butterworth_low':
        h = 1 / (1 + (d / cutoff)**(2 * order))
    elif filter_type == 'butterworth_high':
        h = 1 / (1 + (cutoff / (d + 0.000001))**(2 * order))
    elif filter_type == 'gaussian_low':
        h = np.exp(-(d**2) / (2 * cutoff**2))
    elif filter_type == 'gaussian_high':
        h = 1 - np.exp(-(d**2) / (2 * cutoff**2))
    elif filter_type == 'laplacian':
        h = -4 * np.pi**2 * (u**2 + v**2)
        h = h / np.max(np.abs(h))
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    filtered_fshift = fshift * h
    filtered_f = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.fft.ifft2(filtered_f)
    filtered_image = np.abs(filtered_image)
    filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image))
    return filtered_image, h

def evaluate_performance(original, processed, title):
    psnr_value = psnr(original, processed)
    ssim_value = ssim(original, processed, data_range=1.0)
    print(f"{title}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")
    return psnr_value, ssim_value

def main():
    try:
        print("Attempting to load image.png...")
        image_path = 'image.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception(f"Failed to load image from '{image_path}'. The file might not exist or is not readable.")
        print(f"Successfully loaded image with shape: {image.shape}")
        image = image / 255.0
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Falling back to sample image...")
        image = rgb2gray(data.astronaut())
        print(f"Loaded sample image with shape: {image.shape}")
    h, w = image.shape
    x, y = w//2 - 10, h//2 - 10
    subset = extract_subset(image, x, y)
    noisy_subset = add_noise(subset, 'gaussian')
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(subset, cmap='gray')
    plt.title('Original Subset (20x20)')
    plt.axis('off')
    plt.subplot(3, 3, 2)
    plt.imshow(noisy_subset, cmap='gray')
    plt.title('Noisy Subset')
    plt.axis('off')
    filters = [
        ('Ideal Lowpass', 'ideal_low', 5),
        ('Butterworth Lowpass', 'butterworth_low', 5),
        ('Gaussian Lowpass', 'gaussian_low', 5),
        ('Ideal Highpass', 'ideal_high', 2),
        ('Butterworth Highpass', 'butterworth_high', 2),
        ('Gaussian Highpass', 'gaussian_high', 2),
        ('Laplacian Sharpening', 'laplacian', None)
    ]
    results = []
    performance_metrics = []
    for i, (title, filter_type, cutoff) in enumerate(filters):
        if cutoff is None:
            filtered, h = apply_frequency_filter(noisy_subset, filter_type)
        else:
            filtered, h = apply_frequency_filter(noisy_subset, filter_type, cutoff)
        results.append((filtered, h))
        plt.subplot(3, 3, i+3)
        plt.imshow(filtered, cmap='gray')
        plt.title(title)
        plt.axis('off')
        metrics = evaluate_performance(subset, filtered, title)
        performance_metrics.append((title, metrics))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar([item[0] for item in performance_metrics], 
            [item[1][0] for item in performance_metrics])
    plt.title('PSNR Comparison')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45, ha='right')
    plt.subplot(1, 2, 2)
    plt.bar([item[0] for item in performance_metrics], 
            [item[1][1] for item in performance_metrics])
    plt.title('SSIM Comparison')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show() 
    plt.figure(figsize=(15, 10))
    for i, (title, filter_type, _) in enumerate(filters):
        plt.subplot(2, 4, i+1)
        plt.imshow(results[i][1], cmap='gray')
        plt.title(f"{title} Filter")
        plt.axis('off') 
    plt.tight_layout()
    plt.show()
    print("\nConclusion:")
    print("1. Smoothing filters (lowpass) are effective at removing noise but may blur edges")
    print("2. Sharpening filters (highpass, Laplacian) enhance edges but may amplify noise")
    print("3. Gaussian filters generally provide a better balance between noise removal and detail preservation")
    print("4. Butterworth filters offer a compromise between ideal and Gaussian filters")
if __name__ == "__main__":
    main()