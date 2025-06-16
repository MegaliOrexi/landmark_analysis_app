import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any
import os


class TimeOfDayClassifier:
    """
    Class for classifying images as daytime or nighttime using multiple color space approaches.
    """
    
    def __init__(self):
        """
        Initialize the time of day classifier with default thresholds.
        """
        # Default thresholds for different methods
        self.thresholds = {
            'hsv_value': 80.0,  # Threshold for mean V channel in HSV
            'gray_mean': 100.0,  # Threshold for mean grayscale intensity
            'lab_lightness': 50.0,  # Threshold for mean L channel in LAB
            'bright_pixel_ratio': 0.5,  # Ratio of bright pixels (V > 100) for HSV
            'dark_pixel_ratio': 0.6  # Ratio of dark pixels (V < 50) for HSV
        }
        
        # Weights for ensemble method (sum to 1.0)
        self.method_weights = {
            'hsv_value': 0.3,
            'gray_mean': 0.2,
            'lab_lightness': 0.3,
            'bright_pixel_ratio': 0.1,
            'dark_pixel_ratio': 0.1
        }
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set custom thresholds for classification methods.
        
        Args:
            thresholds: Dictionary mapping method names to threshold values
        """
        for method, threshold in thresholds.items():
            if method in self.thresholds:
                self.thresholds[method] = threshold
            else:
                print(f"Warning: Unknown method '{method}'. Threshold not set.")
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for ensemble classification.
        
        Args:
            weights: Dictionary mapping method names to weight values (should sum to 1.0)
        """
        # Check if weights sum to approximately 1.0
        if abs(sum(weights.values()) - 1.0) > 0.001:
            print(f"Warning: Weights sum to {sum(weights.values())}, not 1.0. Normalizing.")
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        for method, weight in weights.items():
            if method in self.method_weights:
                self.method_weights[method] = weight
            else:
                print(f"Warning: Unknown method '{method}'. Weight not set.")
    
    def _classify_hsv_value(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify based on mean value (brightness) in HSV color space.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract V channel (brightness)
        v_channel = hsv[:, :, 2]
        
        # Calculate mean brightness
        mean_v = np.mean(v_channel)
        
        # Calculate histogram for visualization
        hist_v = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
        
        # Classify based on threshold
        if mean_v >= self.thresholds['hsv_value']:
            classification = 'day'
            # Higher mean_v means higher confidence for day
            confidence = min(1.0, (mean_v - self.thresholds['hsv_value']) / 100.0 + 0.5)
        else:
            classification = 'night'
            # Lower mean_v means higher confidence for night
            confidence = min(1.0, (self.thresholds['hsv_value'] - mean_v) / 100.0 + 0.5)
        
        # Return metrics for analysis
        metrics = {
            'mean_v': mean_v,
            'histogram_v': hist_v,
            'threshold': self.thresholds['hsv_value']
        }
        
        return classification, confidence, metrics
    
    def _classify_grayscale(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify based on mean intensity in grayscale.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean intensity
        mean_gray = np.mean(gray)
        
        # Calculate histogram for visualization
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Classify based on threshold
        if mean_gray >= self.thresholds['gray_mean']:
            classification = 'day'
            # Higher mean_gray means higher confidence for day
            confidence = min(1.0, (mean_gray - self.thresholds['gray_mean']) / 100.0 + 0.5)
        else:
            classification = 'night'
            # Lower mean_gray means higher confidence for night
            confidence = min(1.0, (self.thresholds['gray_mean'] - mean_gray) / 100.0 + 0.5)
        
        # Return metrics for analysis
        metrics = {
            'mean_gray': mean_gray,
            'histogram_gray': hist_gray,
            'threshold': self.thresholds['gray_mean']
        }
        
        return classification, confidence, metrics
    
    def _classify_lab_lightness(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify based on mean lightness in LAB color space.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract L channel (lightness)
        l_channel = lab[:, :, 0]
        
        # Calculate mean lightness (L ranges from 0 to 255 in OpenCV)
        mean_l = np.mean(l_channel)
        
        # Calculate histogram for visualization
        hist_l = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
        
        # Classify based on threshold
        if mean_l >= self.thresholds['lab_lightness']:
            classification = 'day'
            # Higher mean_l means higher confidence for day
            confidence = min(1.0, (mean_l - self.thresholds['lab_lightness']) / 100.0 + 0.5)
        else:
            classification = 'night'
            # Lower mean_l means higher confidence for night
            confidence = min(1.0, (self.thresholds['lab_lightness'] - mean_l) / 100.0 + 0.5)
        
        # Return metrics for analysis
        metrics = {
            'mean_l': mean_l,
            'histogram_l': hist_l,
            'threshold': self.thresholds['lab_lightness']
        }
        
        return classification, confidence, metrics
    
    def _classify_bright_pixel_ratio(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify based on ratio of bright pixels in HSV color space.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract V channel (brightness)
        v_channel = hsv[:, :, 2]
        
        # Calculate ratio of bright pixels (V > 100)
        bright_pixels = np.sum(v_channel > 100)
        total_pixels = v_channel.size
        bright_ratio = bright_pixels / total_pixels
        
        # Classify based on threshold
        if bright_ratio >= self.thresholds['bright_pixel_ratio']:
            classification = 'day'
            # Higher bright_ratio means higher confidence for day
            confidence = min(1.0, (bright_ratio - self.thresholds['bright_pixel_ratio']) / 0.3 + 0.5)
        else:
            classification = 'night'
            # Lower bright_ratio means higher confidence for night
            confidence = min(1.0, (self.thresholds['bright_pixel_ratio'] - bright_ratio) / 0.3 + 0.5)
        
        # Return metrics for analysis
        metrics = {
            'bright_ratio': bright_ratio,
            'threshold': self.thresholds['bright_pixel_ratio']
        }
        
        return classification, confidence, metrics
    
    def _classify_dark_pixel_ratio(self, image: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify based on ratio of dark pixels in HSV color space.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract V channel (brightness)
        v_channel = hsv[:, :, 2]
        
        # Calculate ratio of dark pixels (V < 50)
        dark_pixels = np.sum(v_channel < 50)
        total_pixels = v_channel.size
        dark_ratio = dark_pixels / total_pixels
        
        # Classify based on threshold
        if dark_ratio >= self.thresholds['dark_pixel_ratio']:
            classification = 'night'
            # Higher dark_ratio means higher confidence for night
            confidence = min(1.0, (dark_ratio - self.thresholds['dark_pixel_ratio']) / 0.3 + 0.5)
        else:
            classification = 'day'
            # Lower dark_ratio means higher confidence for day
            confidence = min(1.0, (self.thresholds['dark_pixel_ratio'] - dark_ratio) / 0.3 + 0.5)
        
        # Return metrics for analysis
        metrics = {
            'dark_ratio': dark_ratio,
            'threshold': self.thresholds['dark_pixel_ratio']
        }
        
        return classification, confidence, metrics
    
    def classify(self, image: np.ndarray, method: str = 'ensemble') -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify an image as daytime or nighttime.
        
        Args:
            image: Input image (BGR format)
            method: Classification method ('hsv_value', 'gray_mean', 'lab_lightness', 
                    'bright_pixel_ratio', 'dark_pixel_ratio', or 'ensemble')
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Dictionary to store results from all methods
        all_results = {}
        
        # Run all methods
        all_results['hsv_value'] = self._classify_hsv_value(image)
        all_results['gray_mean'] = self._classify_grayscale(image)
        all_results['lab_lightness'] = self._classify_lab_lightness(image)
        all_results['bright_pixel_ratio'] = self._classify_bright_pixel_ratio(image)
        all_results['dark_pixel_ratio'] = self._classify_dark_pixel_ratio(image)
        
        # If a specific method is requested, return its result
        if method != 'ensemble' and method in all_results:
            return all_results[method]
        
        # For ensemble method, combine results from all methods
        day_score = 0.0
        night_score = 0.0
        
        for method_name, (classification, confidence, _) in all_results.items():
            weight = self.method_weights.get(method_name, 0.0)
            if classification == 'day':
                day_score += weight * confidence
            else:
                night_score += weight * confidence
        
        # Determine final classification and confidence
        if day_score >= night_score:
            classification = 'day'
            confidence = day_score / (day_score + night_score) if (day_score + night_score) > 0 else 0.5
        else:
            classification = 'night'
            confidence = night_score / (day_score + night_score) if (day_score + night_score) > 0 else 0.5
        
        # Compile metrics from all methods
        metrics = {
            'day_score': day_score,
            'night_score': night_score,
            'method_results': all_results
        }
        
        return classification, confidence, metrics
    
    def visualize_classification(self, image: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize the classification results with histograms and metrics.
        
        Args:
            image: Input image (BGR format)
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image (or original image if conversion fails)
        """
        # Get classification results from all methods
        classification, confidence, metrics = self.classify(image)
        
        # Create figure for visualization
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display original image
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title(f"Classification: {classification.upper()} (Confidence: {confidence:.2f})")
        axs[0, 0].axis('off')
        
        # Display HSV value histogram
        hsv_result = metrics['method_results']['hsv_value']
        axs[0, 1].plot(hsv_result[2]['histogram_v'])
        axs[0, 1].axvline(x=self.thresholds['hsv_value'], color='r', linestyle='--')
        axs[0, 1].set_title(f"HSV Value (Mean: {hsv_result[2]['mean_v']:.2f}, Threshold: {self.thresholds['hsv_value']})")
        axs[0, 1].set_xlim([0, 256])
        
        # Display grayscale histogram
        gray_result = metrics['method_results']['gray_mean']
        axs[0, 2].plot(gray_result[2]['histogram_gray'])
        axs[0, 2].axvline(x=self.thresholds['gray_mean'], color='r', linestyle='--')
        axs[0, 2].set_title(f"Grayscale (Mean: {gray_result[2]['mean_gray']:.2f}, Threshold: {self.thresholds['gray_mean']})")
        axs[0, 2].set_xlim([0, 256])
        
        # Display LAB lightness histogram
        lab_result = metrics['method_results']['lab_lightness']
        axs[1, 0].plot(lab_result[2]['histogram_l'])
        axs[1, 0].axvline(x=self.thresholds['lab_lightness'], color='r', linestyle='--')
        axs[1, 0].set_title(f"LAB Lightness (Mean: {lab_result[2]['mean_l']:.2f}, Threshold: {self.thresholds['lab_lightness']})")
        axs[1, 0].set_xlim([0, 256])
        
        # Display bright pixel ratio
        bright_result = metrics['method_results']['bright_pixel_ratio']
        axs[1, 1].bar(['Bright', 'Not Bright'], 
                    [bright_result[2]['bright_ratio'], 1 - bright_result[2]['bright_ratio']])
        axs[1, 1].axhline(y=self.thresholds['bright_pixel_ratio'], color='r', linestyle='--')
        axs[1, 1].set_title(f"Bright Pixel Ratio: {bright_result[2]['bright_ratio']:.2f}")
        axs[1, 1].set_ylim([0, 1])
        
        # Display dark pixel ratio
        dark_result = metrics['method_results']['dark_pixel_ratio']
        axs[1, 2].bar(['Dark', 'Not Dark'], 
                    [dark_result[2]['dark_ratio'], 1 - dark_result[2]['dark_ratio']])
        axs[1, 2].axhline(y=self.thresholds['dark_pixel_ratio'], color='r', linestyle='--')
        axs[1, 2].set_title(f"Dark Pixel Ratio: {dark_result[2]['dark_ratio']:.2f}")
        axs[1, 2].set_ylim([0, 1])
        
        # Adjust layout
        plt.tight_layout()
        
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
    
    def calibrate_thresholds(self, day_images: List[np.ndarray], night_images: List[np.ndarray]) -> Dict[str, float]:
        """
        Calibrate thresholds based on labeled training data.
        
        Args:
            day_images: List of daytime images
            night_images: List of nighttime images
            
        Returns:
            Dictionary of calibrated thresholds
        """
        # Collect metrics for all images
        day_metrics = {
            'hsv_value': [],
            'gray_mean': [],
            'lab_lightness': [],
            'bright_pixel_ratio': [],
            'dark_pixel_ratio': []
        }
        
        night_metrics = {
            'hsv_value': [],
            'gray_mean': [],
            'lab_lightness': [],
            'bright_pixel_ratio': [],
            'dark_pixel_ratio': []
        }
        
        # Process day images
        for image in day_images:
            # HSV value
            _, _, metrics = self._classify_hsv_value(image)
            day_metrics['hsv_value'].append(metrics['mean_v'])
            
            # Grayscale
            _, _, metrics = self._classify_grayscale(image)
            day_metrics['gray_mean'].append(metrics['mean_gray'])
            
            # LAB lightness
            _, _, metrics = self._classify_lab_lightness(image)
            day_metrics['lab_lightness'].append(metrics['mean_l'])
            
            # Bright pixel ratio
            _, _, metrics = self._classify_bright_pixel_ratio(image)
            day_metrics['bright_pixel_ratio'].append(metrics['bright_ratio'])
            
            # Dark pixel ratio
            _, _, metrics = self._classify_dark_pixel_ratio(image)
            day_metrics['dark_pixel_ratio'].append(metrics['dark_ratio'])
        
        # Process night images
        for image in night_images:
            # HSV value
            _, _, metrics = self._classify_hsv_value(image)
            night_metrics['hsv_value'].append(metrics['mean_v'])
            
            # Grayscale
            _, _, metrics = self._classify_grayscale(image)
            night_metrics['gray_mean'].append(metrics['mean_gray'])
            
            # LAB lightness
            _, _, metrics = self._classify_lab_lightness(image)
            night_metrics['lab_lightness'].append(metrics['mean_l'])
            
            # Bright pixel ratio
            _, _, metrics = self._classify_bright_pixel_ratio(image)
            night_metrics['bright_pixel_ratio'].append(metrics['bright_ratio'])
            
            # Dark pixel ratio
            _, _, metrics = self._classify_dark_pixel_ratio(image)
            night_metrics['dark_pixel_ratio'].append(metrics['dark_ratio'])
        
        # Calculate optimal thresholds
        calibrated_thresholds = {}
        
        # For each metric, find the threshold that maximizes classification accuracy
        for metric in day_metrics.keys():
            day_values = np.array(day_metrics[metric])
            night_values = np.array(night_metrics[metric])
            
            # Skip if no data
            if len(day_values) == 0 or len(night_values) == 0:
                continue
            
            # For dark_pixel_ratio, night images should have higher values
            if metric == 'dark_pixel_ratio':
                # Find optimal threshold
                min_val = min(np.min(day_values), np.min(night_values))
                max_val = max(np.max(day_values), np.max(night_values))
                
                best_threshold = min_val
                best_accuracy = 0.0
                
                for threshold in np.linspace(min_val, max_val, 100):
                    # Count correct classifications
                    day_correct = np.sum(day_values < threshold)
                    night_correct = np.sum(night_values >= threshold)
                    
                    # Calculate accuracy
                    accuracy = (day_correct + night_correct) / (len(day_values) + len(night_values))
                    
                    # Update best threshold
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                
                calibrated_thresholds[metric] = best_threshold
            else:
                # For other metrics, day images should have higher values
                # Find optimal threshold
                min_val = min(np.min(day_values), np.min(night_values))
                max_val = max(np.max(day_values), np.max(night_values))
                
                best_threshold = min_val
                best_accuracy = 0.0
                
                for threshold in np.linspace(min_val, max_val, 100):
                    # Count correct classifications
                    day_correct = np.sum(day_values >= threshold)
                    night_correct = np.sum(night_values < threshold)
                    
                    # Calculate accuracy
                    accuracy = (day_correct + night_correct) / (len(day_values) + len(night_values))
                    
                    # Update best threshold
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                
                calibrated_thresholds[metric] = best_threshold
        
        # Update thresholds
        self.set_thresholds(calibrated_thresholds)
        
        return calibrated_thresholds


# Example usage:
if __name__ == "__main__":
    # Initialize classifier
    classifier = TimeOfDayClassifier()
    
    # Test on sample images
    sample_dir = "../data/images/time_of_day"  # Update with your image directory
    if os.path.exists(sample_dir):
        # Process day and night sample images
        day_dir = os.path.join(sample_dir, "day")
        night_dir = os.path.join(sample_dir, "night")
        
        day_images = []
        night_images = []
        
        # Load day images
        if os.path.exists(day_dir):
            for filename in os.listdir(day_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(day_dir, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        day_images.append(image)
        
        # Load night images
        if os.path.exists(night_dir):
            for filename in os.listdir(night_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(night_dir, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        night_images.append(image)
        
        # Calibrate thresholds if enough images are available
        if len(day_images) > 0 and len(night_images) > 0:
            print("Calibrating thresholds...")
            calibrated_thresholds = classifier.calibrate_thresholds(day_images, night_images)
            print(f"Calibrated thresholds: {calibrated_thresholds}")
        
        # Test on a sample image
        test_image_path = "../data/images/test.jpg"  # Update with your test image path
        if os.path.exists(test_image_path):
            # Load image
            test_image = cv2.imread(test_image_path)
            
            # Classify using different methods
            for method in ['hsv_value', 'gray_mean', 'lab_lightness', 'bright_pixel_ratio', 'dark_pixel_ratio', 'ensemble']:
                classification, confidence, _ = classifier.classify(test_image, method)
                print(f"Method: {method}, Classification: {classification}, Confidence: {confidence:.2f}")
            
            # Visualize classification
            vis_image = classifier.visualize_classification(test_image)
            
            # Display visualization
            cv2.imshow("Time of Day Classification", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Test image not found: {test_image_path}")
    else:
        print(f"Sample directory not found: {sample_dir}")
