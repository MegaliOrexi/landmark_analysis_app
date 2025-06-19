import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any
import os


class ImageQualityAssessor:
    """
    Class for assessing image quality and suggesting enhancements.
    """
    
    def __init__(self):
        """
        Initialize the image quality assessor with default thresholds.
        """
        # Default thresholds for different quality metrics
        self.thresholds = {
            'blur': 100.0,  # Threshold for Laplacian variance (lower means more blurry)
            'brightness_low': 50.0,  # Lower threshold for mean brightness (0-255)
            'brightness_high': 200.0,  # Upper threshold for mean brightness (0-255)
            'contrast_low': 40.0,  # Lower threshold for standard deviation of pixel values
            'saturation': 0.05,  # Threshold for percentage of saturated pixels
            'noise': 5.0  # Threshold for estimated noise level
        }
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set custom thresholds for quality assessment.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
        """
        for metric, threshold in thresholds.items():
            if metric in self.thresholds:
                self.thresholds[metric] = threshold
            else:
                print(f"Warning: Unknown metric '{metric}'. Threshold not set.")
    
    def detect_blur(self, image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Detect if an image is blurry using Laplacian variance.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (is_blurry, laplacian_variance, laplacian_visualization)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Compute variance of Laplacian
        laplacian_var = laplacian.var()
        
        # Determine if image is blurry
        is_blurry = laplacian_var < self.thresholds['blur']
        
        # Create visualization of Laplacian
        laplacian_vis = cv2.convertScaleAbs(laplacian)
        
        return is_blurry, laplacian_var, laplacian_vis
    
    def analyze_brightness(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyze image brightness.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (brightness_status, mean_brightness, metrics)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute mean brightness
        mean_brightness = np.mean(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate percentage of dark and bright pixels
        dark_pixels = np.sum(gray < 30) / gray.size
        bright_pixels = np.sum(gray > 220) / gray.size
        
        # Determine brightness status
        if mean_brightness < self.thresholds['brightness_low']:
            brightness_status = 'too_dark'
        elif mean_brightness > self.thresholds['brightness_high']:
            brightness_status = 'too_bright'
        else:
            brightness_status = 'good'
        
        # Return metrics
        metrics = {
            'mean_brightness': mean_brightness,
            'dark_pixel_ratio': dark_pixels,
            'bright_pixel_ratio': bright_pixels,
            'histogram': hist
        }
        
        return brightness_status, mean_brightness, metrics
    
    def analyze_contrast(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyze image contrast.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (contrast_status, std_dev, metrics)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute standard deviation as a measure of contrast
        std_dev = np.std(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate histogram spread (distance between 5th and 95th percentiles)
        hist_cumsum = np.cumsum(hist) / np.sum(hist)
        p5_idx = np.argmax(hist_cumsum >= 0.05)
        p95_idx = np.argmax(hist_cumsum >= 0.95)
        hist_spread = p95_idx - p5_idx
        
        # Determine contrast status
        if std_dev < self.thresholds['contrast_low']:
            contrast_status = 'low_contrast'
        else:
            contrast_status = 'good'
        
        # Return metrics
        metrics = {
            'std_dev': std_dev,
            'histogram': hist,
            'histogram_spread': hist_spread
        }
        
        return contrast_status, std_dev, metrics
    
    def detect_saturation(self, image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Detect if an image has saturated pixels.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (has_saturation, saturation_ratio, saturation_mask)
        """
        # Check for saturated pixels (0 or 255 in any channel)
        if len(image.shape) == 3:
            # Check each channel
            saturated_b = np.logical_or(image[:,:,0] == 0, image[:,:,0] == 255)
            saturated_g = np.logical_or(image[:,:,1] == 0, image[:,:,1] == 255)
            saturated_r = np.logical_or(image[:,:,2] == 0, image[:,:,2] == 255)
            
            # Combine masks
            saturated_mask = np.logical_or(saturated_b, np.logical_or(saturated_g, saturated_r))
        else:
            # Grayscale image
            saturated_mask = np.logical_or(image == 0, image == 255)
        
        # Calculate ratio of saturated pixels
        saturation_ratio = np.sum(saturated_mask) / saturated_mask.size
        
        # Determine if image has significant saturation
        has_saturation = saturation_ratio > self.thresholds['saturation']
        
        # Create visualization of saturated pixels
        saturation_vis = np.zeros_like(image)
        if len(image.shape) == 3:
            saturation_vis[saturated_mask] = [0, 0, 255]  # Red for saturated pixels
        else:
            saturation_vis[saturated_mask] = 255
        
        return has_saturation, saturation_ratio, saturation_vis
    
    def estimate_noise(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimate the noise level in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (noise_level, noise_visualization)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray, 5)
        
        # Calculate difference between original and denoised image
        noise = cv2.absdiff(gray, denoised)
        
        # Estimate noise level as mean of absolute differences
        noise_level = np.mean(noise)
        
        return noise_level, noise
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment on an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing assessment results and metrics
        """
        # Make a copy of the image
        img_copy = image.copy()
        
        # Detect blur
        is_blurry, laplacian_var, laplacian_vis = self.detect_blur(img_copy)
        
        # Analyze brightness
        brightness_status, mean_brightness, brightness_metrics = self.analyze_brightness(img_copy)
        
        # Analyze contrast
        contrast_status, std_dev, contrast_metrics = self.analyze_contrast(img_copy)
        
        # Detect saturation
        has_saturation, saturation_ratio, saturation_vis = self.detect_saturation(img_copy)
        
        # Estimate noise
        noise_level, noise_vis = self.estimate_noise(img_copy)
        
        # Determine overall quality
        quality_issues = []
        if is_blurry:
            quality_issues.append('blurry')
        if brightness_status != 'good':
            quality_issues.append(brightness_status)
        if contrast_status != 'good':
            quality_issues.append(contrast_status)
        if has_saturation:
            quality_issues.append('saturation')
        if noise_level > self.thresholds['noise']:
            quality_issues.append('noisy')
        
        # Generate improvement suggestions
        suggestions = []
        if is_blurry:
            suggestions.append('Capture with better focus or apply sharpening')
        if brightness_status == 'too_dark':
            suggestions.append('Increase exposure or apply brightness enhancement')
        elif brightness_status == 'too_bright':
            suggestions.append('Decrease exposure or apply brightness reduction')
        if contrast_status == 'low_contrast':
            suggestions.append('Apply contrast enhancement or histogram equalization')
        if has_saturation:
            suggestions.append('Adjust exposure to avoid saturated pixels')
        if noise_level > self.thresholds['noise']:
            suggestions.append('Use lower ISO setting or apply noise reduction')
        
        # Compile results
        assessment = {
            'overall_quality': 'good' if not quality_issues else 'needs_improvement',
            'quality_issues': quality_issues,
            'improvement_suggestions': suggestions,
            'metrics': {
                'blur': {
                    'is_blurry': is_blurry,
                    'laplacian_variance': laplacian_var,
                    'threshold': self.thresholds['blur']
                },
                'brightness': {
                    'status': brightness_status,
                    'mean_brightness': mean_brightness,
                    'dark_pixel_ratio': brightness_metrics['dark_pixel_ratio'],
                    'bright_pixel_ratio': brightness_metrics['bright_pixel_ratio'],
                    'low_threshold': self.thresholds['brightness_low'],
                    'high_threshold': self.thresholds['brightness_high']
                },
                'contrast': {
                    'status': contrast_status,
                    'std_dev': std_dev,
                    'histogram_spread': contrast_metrics['histogram_spread'],
                    'threshold': self.thresholds['contrast_low']
                },
                'saturation': {
                    'has_saturation': has_saturation,
                    'saturation_ratio': saturation_ratio,
                    'threshold': self.thresholds['saturation']
                },
                'noise': {
                    'noise_level': noise_level,
                    'threshold': self.thresholds['noise']
                }
            },
            'visualizations': {
                'laplacian': laplacian_vis,
                'saturation_mask': saturation_vis,
                'noise': noise_vis
            }
        }
        
        return assessment
    
    def highlight_issues(self, image: np.ndarray, assessment: Dict[str, Any]) -> np.ndarray:
        """
        Highlight areas in the image that need improvement.
        
        Args:
            image: Input image (BGR format)
            assessment: Assessment results from assess_quality()
            
        Returns:
            Image with highlighted issues
        """
        # Make a copy of the image
        highlighted = image.copy()
        
        # Ensure we're working with a color image
        if len(highlighted.shape) == 2:
            highlighted = cv2.cvtColor(highlighted, cv2.COLOR_GRAY2BGR)
        
        # Create an overlay for highlighting issues
        overlay = np.zeros_like(highlighted)
        
        # Highlight blurry regions
        if 'blurry' in assessment['quality_issues']:
            # Use Laplacian to identify edges (low values indicate blur)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            abs_laplacian = cv2.convertScaleAbs(laplacian)
            
            # Threshold to find areas with low edge response
            _, blur_mask = cv2.threshold(abs_laplacian, 10, 255, cv2.THRESH_BINARY_INV)
            
            # Apply blue highlight for blurry areas
            overlay[blur_mask > 0] = [255, 0, 0]  # Blue
        
        # Highlight brightness issues
        if 'too_dark' in assessment['quality_issues']:
            # Create mask for dark areas
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Apply green highlight for dark areas
            overlay[dark_mask > 0] = [0, 255, 0]  # Green
        
        elif 'too_bright' in assessment['quality_issues']:
            # Create mask for bright areas
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Apply yellow highlight for bright areas
            overlay[bright_mask > 0] = [0, 255, 255]  # Yellow
        
        # Highlight saturation issues
        if 'saturation' in assessment['quality_issues']:
            # Get saturation visualization from assessment
            saturation_vis = assessment['visualizations']['saturation_mask']
            
            # Create proper mask - check if saturation_vis is grayscale or color
            if len(saturation_vis.shape) == 3:
                # If color, convert to grayscale to create mask
                saturation_mask = cv2.cvtColor(saturation_vis, cv2.COLOR_BGR2GRAY) > 0
            else:
                # If grayscale, use directly
                saturation_mask = saturation_vis > 0
            
            # Apply red highlight for saturated pixels
            overlay[saturation_mask] = [0, 0, 255]  # Red
        
        # Blend overlay with original image
        alpha = 0.3  # Transparency factor
        highlighted = cv2.addWeighted(highlighted, 1 - alpha, overlay, alpha, 0)
        
        return highlighted
    
    def visualize_assessment(self, image: np.ndarray, assessment: Dict[str, Any], 
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Create a visualization of the quality assessment results.
        
        Args:
            image: Input image (BGR format)
            assessment: Assessment results from assess_quality()
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        # Create figure for visualization
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display original image
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')
        
        # Display highlighted issues
        highlighted = self.highlight_issues(image, assessment)
        axs[0, 1].imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Highlighted Issues")
        axs[0, 1].axis('off')
        
        # Display Laplacian visualization (edges)
        axs[0, 2].imshow(assessment['visualizations']['laplacian'], cmap='gray')
        axs[0, 2].set_title(f"Laplacian (Blur: {assessment['metrics']['blur']['laplacian_variance']:.2f})")
        axs[0, 2].axis('off')
        
        # Display brightness histogram
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axs[1, 0].plot(hist)
        axs[1, 0].axvline(x=assessment['metrics']['brightness']['low_threshold'], color='r', linestyle='--')
        axs[1, 0].axvline(x=assessment['metrics']['brightness']['high_threshold'], color='r', linestyle='--')
        axs[1, 0].set_title(f"Brightness: {assessment['metrics']['brightness']['mean_brightness']:.2f}")
        axs[1, 0].set_xlim([0, 256])
        
        # Display saturation mask
        axs[1, 1].imshow(assessment['visualizations']['saturation_mask'])
        axs[1, 1].set_title(f"Saturation: {assessment['metrics']['saturation']['saturation_ratio']:.2%}")
        axs[1, 1].axis('off')
        
        # Display noise visualization
        axs[1, 2].imshow(assessment['visualizations']['noise'], cmap='hot')
        axs[1, 2].set_title(f"Noise Level: {assessment['metrics']['noise']['noise_level']:.2f}")
        axs[1, 2].axis('off')
        
        # Add text for quality issues and suggestions
        # issues_text = "Quality Issues: " + ", ".join(assessment['quality_issues']) if assessment['quality_issues'] else "No issues detected"
        suggestions_text = "Suggestions:\n" + "\n".join(assessment['improvement_suggestions']) if assessment['improvement_suggestions'] else "No improvements needed"
        
        # fig.text(0.5, 0.02, issues_text, ha='center', fontsize=12, color='red' if assessment['quality_issues'] else 'green')
        fig.text(0.5, 0.01, suggestions_text, ha='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save if output path is provided
        if output_path is not None:
            plt.savefig(output_path)
        
        # Convert figure to image with error handling
        try:
            fig.canvas.draw()
            vis_image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # ARGB has 4 channels
            # Convert ARGB to RGB
            vis_image = vis_image[:, :, 1:]  # Remove alpha channel
        except (ValueError, AttributeError) as e:
            # If conversion fails, return the original image
            print(f"Warning: Could not convert visualization to array: {e}")
        
        # Close figure to free memory
        plt.close(fig)
        
        return vis_image
    
    def enhance_image(self, image: np.ndarray, assessment: Dict[str, Any]) -> np.ndarray:
        """
        Enhance the image based on quality assessment.
        
        Args:
            image: Input image (BGR format)
            assessment: Assessment results from assess_quality()
            
        Returns:
            Enhanced image
        """
        # Make a copy of the image
        enhanced = image.copy()
        
        # Apply enhancements based on quality issues
        for issue in assessment['quality_issues']:
            if issue == 'blurry':
                # Apply sharpening
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            elif issue == 'too_dark':
                # Apply brightness enhancement
                alpha = 1.5  # Contrast control
                beta = 30    # Brightness control
                enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            elif issue == 'too_bright':
                # Reduce brightness
                alpha = 0.8  # Contrast control
                beta = -20   # Brightness control
                enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            elif issue == 'low_contrast':
                # Apply histogram equalization
                if len(enhanced.shape) == 3:
                    # Convert to YUV and equalize Y channel
                    yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    # Grayscale image
                    enhanced = cv2.equalizeHist(enhanced)
            
            elif issue == 'noisy':
                # Apply noise reduction
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21) if len(enhanced.shape) == 3 else cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return enhanced


# Example usage:
if __name__ == "__main__":
    # Initialize quality assessor
    assessor = ImageQualityAssessor()
    
    # Test on sample images
    sample_dir = "../data/images/quality"  # Update with your image directory
    if os.path.exists(sample_dir):
        # Process each image
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(sample_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    print(f"Assessing quality of {filename}...")
                    
                    # Assess quality
                    assessment = assessor.assess_quality(image)
                    
                    # Print results
                    print(f"Overall quality: {assessment['overall_quality']}")
                    if assessment['quality_issues']:
                        print(f"Quality issues: {', '.join(assessment['quality_issues'])}")
                    if assessment['improvement_suggestions']:
                        print(f"Suggestions: {', '.join(assessment['improvement_suggestions'])}")
                    
                    # Visualize assessment
                    vis_image = assessor.visualize_assessment(image, assessment)
                    
                    # Enhance image if needed
                    if assessment['overall_quality'] != 'good':
                        enhanced = assessor.enhance_image(image, assessment)
                        
                        # Display original and enhanced images
                        comparison = np.hstack((image, enhanced))
                        cv2.imshow("Original vs Enhanced", comparison)
                    else:
                        # Display original image
                        cv2.imshow("Original Image", image)
                    
                    # Display assessment visualization
                    cv2.imshow("Quality Assessment", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        print(f"Sample directory not found: {sample_dir}")
