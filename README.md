# Frequency Domain Image Processing

This project performs various image enhancement techniques in the frequency domain on a 20x20 subset of an image. It implements several filters for smoothing and sharpening, and compares their performance using objective metrics.

## Overview

The code extracts a 20x20 subset from an input image, adds Gaussian noise, and then applies various frequency domain filters:
- Ideal Lowpass
- Butterworth Lowpass
- Gaussian Lowpass
- Ideal Highpass
- Butterworth Highpass
- Gaussian Highpass
- Laplacian Sharpening

Performance evaluation is done using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) metrics.

## Requirements

- Python 3.x
- Libraries: OpenCV (cv2), NumPy, Matplotlib, scikit-image

Install dependencies with:
```bash
pip install opencv-python numpy matplotlib scikit-image
```

## Usage

1. Place your input image in the same directory as the script and name it `image.png`
2. Run the script:
```bash
python app.py
```

If `image.png` is not found, the script will fall back to using a sample image from scikit-image.

## Expected Output

The script generates three visualization figures:

1. **Filter Results**: Shows the original image subset, the noisy subset, and the results of applying each filter
2. **Performance Metrics**: Bar charts comparing PSNR and SSIM values for each filter
3. **Filter Visualizations**: Shows the actual filter masks in the frequency domain

The script also prints performance metrics in the console and concludes with a summary of findings.

## Sample Output Images

When you run the code, place screenshots of your output in this section. You should include:

1. The first figure showing:
   - Original 20x20 subset
   - Noisy subset
   - Results of different filters

2. The performance comparison bar charts

3. The frequency domain filter visualizations

## Understanding the Results

### Smoothing Filters (Lowpass)
- **Ideal Lowpass**: Sharp cutoff frequency, may cause ringing artifacts
- **Butterworth Lowpass**: Smoother transition, reduces ringing
- **Gaussian Lowpass**: Very smooth transition, optimal for Gaussian noise

### Sharpening Filters (Highpass)
- **Ideal Highpass**: Enhances all high frequencies, including noise
- **Butterworth Highpass**: More controlled high-frequency enhancement
- **Gaussian Highpass**: Gradual enhancement of high frequencies
- **Laplacian**: Emphasizes regions of rapid intensity change

### Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better noise suppression
- **SSIM (Structural Similarity Index)**: Values closer to 1 indicate better structural preservation

## Modifying the Code:

You can adjust these parameters to experiment with different results:
- Change the noise type ('gaussian' or 'salt_pepper')
- Modify cutoff frequencies for filters
- Change the order for Butterworth filters

## Troubleshooting

- **Image not loading**: Ensure your image file is named 'image.png' and is in the same directory
- **Poor results**: Try adjusting filter cutoff values or the noise parameters

## Insights on Filter Performance

When analyzing your results, look for:
1. Which filter provides the best PSNR and SSIM?
2. Is there a trade-off between noise reduction and detail preservation?
3. How do the lowpass (smoothing) filters compare to highpass (sharpening) filters?
4. Does the visual appearance match what the metrics suggest?

Generally:
- Lowpass filters are better for noise removal but may blur edges
- Highpass filters enhance details but may amplify noise
- Gaussian filters often provide a good balance between noise removal and detail preservation
- Butterworth filters offer a compromise between ideal and Gaussian filters