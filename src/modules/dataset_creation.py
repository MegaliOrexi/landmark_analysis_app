import cv2
import os
import numpy as np
import json
import shutil
from typing import List, Dict, Tuple, Optional, Any
import uuid
from datetime import datetime


class DatasetCreator:
    """
    Class for creating and managing image datasets with proper tagging.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the dataset creator.
        
        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = base_dir
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Dictionary to store dataset metadata
        self.datasets = {}
        
        # Load existing datasets if any
        self._load_datasets()
    
    def _load_datasets(self) -> None:
        """
        Load metadata for existing datasets.
        """
        metadata_path = os.path.join(self.base_dir, 'datasets_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.datasets = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse dataset metadata file. Starting with empty metadata.")
                self.datasets = {}
    
    def _save_datasets(self) -> None:
        """
        Save metadata for all datasets.
        """
        metadata_path = os.path.join(self.base_dir, 'datasets_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.datasets, f, indent=2)
    
    def create_dataset(self, name: str, dataset_type: str, description: Optional[str] = None) -> str:
        """
        Create a new dataset.
        
        Args:
            name: Name of the dataset
            dataset_type: Type of the dataset (e.g., 'landmarks', 'faces', 'general')
            description: Optional description of the dataset
            
        Returns:
            Dataset ID
        """
        # Generate a unique ID for the dataset
        dataset_id = str(uuid.uuid4())
        
        # Create dataset directory
        dataset_dir = os.path.join(self.base_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            'id': dataset_id,
            'name': name,
            'type': dataset_type,
            'description': description or '',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'image_count': 0,
            'tags': {},
            'classes': []
        }
        
        # Save metadata
        self.datasets[dataset_id] = metadata
        self._save_datasets()
        
        return dataset_id
    
    def add_class(self, dataset_id: str, class_name: str, description: Optional[str] = None) -> None:
        """
        Add a class to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            class_name: Name of the class
            description: Optional description of the class
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Check if class already exists
        if class_name in self.datasets[dataset_id]['classes']:
            print(f"Warning: Class '{class_name}' already exists in dataset")
            return
        
        # Add class
        self.datasets[dataset_id]['classes'].append(class_name)
        
        # Create class directory
        class_dir = os.path.join(self.base_dir, dataset_id, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Update metadata
        self.datasets[dataset_id]['updated_at'] = datetime.now().isoformat()
        self._save_datasets()
    
    def add_image(self, dataset_id: str, image_path: str, class_name: Optional[str] = None, 
                 tags: Optional[Dict[str, Any]] = None, copy: bool = True) -> str:
        """
        Add an image to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            image_path: Path to the image file
            class_name: Optional class name for the image
            tags: Optional dictionary of tags for the image
            copy: Whether to copy the image to the dataset directory (True) or move it (False)
            
        Returns:
            Image ID
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if class exists if provided
        if class_name is not None and class_name not in self.datasets[dataset_id]['classes']:
            raise ValueError(f"Class '{class_name}' not found in dataset")
        
        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())
        
        # Determine target directory
        if class_name is not None:
            target_dir = os.path.join(self.base_dir, dataset_id, class_name)
        else:
            target_dir = os.path.join(self.base_dir, dataset_id)
        
        # Get file extension
        _, ext = os.path.splitext(image_path)
        
        # Create target path
        target_path = os.path.join(target_dir, f"{image_id}{ext}")
        
        # Copy or move image
        if copy:
            shutil.copy2(image_path, target_path)
        else:
            shutil.move(image_path, target_path)
        
        # Create tags if not provided
        if tags is None:
            tags = {}
        
        # Add default tags
        tags['added_at'] = datetime.now().isoformat()
        if class_name is not None:
            tags['class'] = class_name
        
        # Update dataset metadata
        self.datasets[dataset_id]['image_count'] += 1
        self.datasets[dataset_id]['updated_at'] = datetime.now().isoformat()
        self.datasets[dataset_id]['tags'][image_id] = tags
        self._save_datasets()
        
        return image_id
    
    def add_images_from_directory(self, dataset_id: str, directory: str, 
                                class_from_subdirs: bool = True, 
                                tags: Optional[Dict[str, Any]] = None,
                                extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[str]:
        """
        Add multiple images from a directory to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            directory: Directory containing images
            class_from_subdirs: Whether to use subdirectory names as class names
            tags: Optional dictionary of tags to apply to all images
            extensions: List of valid file extensions
            
        Returns:
            List of added image IDs
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Check if directory exists
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Create tags if not provided
        if tags is None:
            tags = {}
        
        # Add default tags
        tags['batch_added_at'] = datetime.now().isoformat()
        
        # List to store added image IDs
        added_image_ids = []
        
        if class_from_subdirs:
            # Process each subdirectory as a class
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                
                if os.path.isdir(subdir_path):
                    # Add class if it doesn't exist
                    if subdir not in self.datasets[dataset_id]['classes']:
                        self.add_class(dataset_id, subdir)
                    
                    # Process each image in the subdirectory
                    for filename in os.listdir(subdir_path):
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            image_path = os.path.join(subdir_path, filename)
                            
                            # Add image with class
                            image_id = self.add_image(dataset_id, image_path, subdir, tags.copy())
                            added_image_ids.append(image_id)
        else:
            # Process all images in the directory
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in extensions):
                    image_path = os.path.join(directory, filename)
                    
                    # Add image without class
                    image_id = self.add_image(dataset_id, image_path, None, tags.copy())
                    added_image_ids.append(image_id)
        
        return added_image_ids
    
    def tag_image(self, dataset_id: str, image_id: str, tags: Dict[str, Any]) -> None:
        """
        Add or update tags for an image.
        
        Args:
            dataset_id: ID of the dataset
            image_id: ID of the image
            tags: Dictionary of tags to add or update
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Check if image exists
        if image_id not in self.datasets[dataset_id]['tags']:
            raise ValueError(f"Image with ID {image_id} not found in dataset")
        
        # Update tags
        self.datasets[dataset_id]['tags'][image_id].update(tags)
        
        # Update metadata
        self.datasets[dataset_id]['updated_at'] = datetime.now().isoformat()
        self._save_datasets()
    
    def get_image_path(self, dataset_id: str, image_id: str) -> str:
        """
        Get the file path for an image.
        
        Args:
            dataset_id: ID of the dataset
            image_id: ID of the image
            
        Returns:
            Path to the image file
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Check if image exists
        if image_id not in self.datasets[dataset_id]['tags']:
            raise ValueError(f"Image with ID {image_id} not found in dataset")
        
        # Get class if available
        class_name = self.datasets[dataset_id]['tags'][image_id].get('class')
        
        # Search for the image file
        if class_name is not None:
            search_dir = os.path.join(self.base_dir, dataset_id, class_name)
        else:
            search_dir = os.path.join(self.base_dir, dataset_id)
        
        # Find the image file
        for filename in os.listdir(search_dir):
            if filename.startswith(image_id):
                return os.path.join(search_dir, filename)
        
        raise FileNotFoundError(f"Image file not found for image ID {image_id}")
    
    def get_images_by_tag(self, dataset_id: str, tag_name: str, tag_value: Any = None) -> List[str]:
        """
        Get images that have a specific tag.
        
        Args:
            dataset_id: ID of the dataset
            tag_name: Name of the tag
            tag_value: Optional value of the tag (if None, just checks for tag existence)
            
        Returns:
            List of image IDs
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # List to store matching image IDs
        matching_images = []
        
        # Check each image
        for image_id, tags in self.datasets[dataset_id]['tags'].items():
            if tag_name in tags:
                if tag_value is None or tags[tag_name] == tag_value:
                    matching_images.append(image_id)
        
        return matching_images
    
    def get_images_by_class(self, dataset_id: str, class_name: str) -> List[str]:
        """
        Get images that belong to a specific class.
        
        Args:
            dataset_id: ID of the dataset
            class_name: Name of the class
            
        Returns:
            List of image IDs
        """
        return self.get_images_by_tag(dataset_id, 'class', class_name)
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Return a copy of the dataset metadata
        return dict(self.datasets[dataset_id])
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets.
        
        Returns:
            List of dictionaries with dataset information
        """
        return [dict(dataset) for dataset in self.datasets.values()]
    
    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Delete dataset directory
        dataset_dir = os.path.join(self.base_dir, dataset_id)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        
        # Remove from metadata
        del self.datasets[dataset_id]
        self._save_datasets()
    
    def export_dataset(self, dataset_id: str, output_dir: str, format: str = 'directory') -> str:
        """
        Export a dataset to a directory.
        
        Args:
            dataset_id: ID of the dataset
            output_dir: Directory to export to
            format: Export format ('directory' or 'zip')
            
        Returns:
            Path to the exported dataset
        """
        # Check if dataset exists
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset information
        dataset_info = self.get_dataset_info(dataset_id)
        
        # Create export directory
        export_dir = os.path.join(output_dir, f"{dataset_info['name']}_{dataset_id}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Export metadata
        with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Export images
        for class_name in dataset_info['classes']:
            # Create class directory
            class_dir = os.path.join(export_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Get images for this class
            image_ids = self.get_images_by_class(dataset_id, class_name)
            
            # Copy images
            for image_id in image_ids:
                try:
                    image_path = self.get_image_path(dataset_id, image_id)
                    _, ext = os.path.splitext(image_path)
                    
                    # Create a more descriptive filename
                    tags = dataset_info['tags'][image_id]
                    timestamp = tags.get('added_at', '').replace(':', '-').replace('.', '-')
                    
                    # Use original filename if available
                    if 'original_filename' in tags:
                        new_filename = f"{tags['original_filename']}_{image_id[:8]}{ext}"
                    else:
                        new_filename = f"{class_name}_{timestamp}_{image_id[:8]}{ext}"
                    
                    # Copy image
                    shutil.copy2(image_path, os.path.join(class_dir, new_filename))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Warning: Could not export image {image_id}: {e}")
        
        # Create a zip file if requested
        if format == 'zip':
            import zipfile
            
            # Create zip file
            zip_path = f"{export_dir}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
            
            # Delete the temporary directory
            shutil.rmtree(export_dir)
            
            return zip_path
        
        return export_dir


class ImageTagger:
    """
    Class for automatically tagging images with various attributes.
    """
    
    def __init__(self):
        """
        Initialize the image tagger.
        """
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize ORB detector for feature extraction
        self.orb = cv2.ORB_create(nfeatures=1000)
    
    def tag_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect and tag faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with face-related tags
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Create tags
        tags = {
            'face_count': len(faces),
            'has_faces': len(faces) > 0,
            'face_locations': faces.tolist() if len(faces) > 0 else []
        }
        
        return tags
    
    def tag_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract and tag color information from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with color-related tags
        """
        # Convert to different color spaces
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate mean color values
            mean_bgr = np.mean(image, axis=(0, 1)).tolist()
            mean_hsv = np.mean(hsv, axis=(0, 1)).tolist()
            mean_lab = np.mean(lab, axis=(0, 1)).tolist()
            
            # Calculate dominant colors using K-means clustering
            pixels = image.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 5  # Number of dominant colors
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Count occurrences of each cluster
            counts = np.bincount(labels.flatten())
            
            # Sort clusters by occurrence
            sorted_indices = np.argsort(counts)[::-1]
            sorted_centers = centers[sorted_indices].astype(int).tolist()
            sorted_percentages = (counts[sorted_indices] / len(labels) * 100).tolist()
            
            # Create tags
            tags = {
                'mean_bgr': mean_bgr,
                'mean_hsv': mean_hsv,
                'mean_lab': mean_lab,
                'dominant_colors': sorted_centers,
                'dominant_color_percentages': sorted_percentages
            }
        else:
            # Grayscale image
            mean_gray = np.mean(image)
            
            # Create tags
            tags = {
                'mean_gray': float(mean_gray),
                'is_grayscale': True
            }
        
        return tags
    
    def tag_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract and tag feature information from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with feature-related tags
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect ORB keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Calculate edge density using Canny edge detector
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate image entropy as a measure of texture
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Create tags
        tags = {
            'keypoint_count': len(keypoints),
            'edge_density': float(edge_density),
            'entropy': float(entropy)
        }
        
        return tags
    
    def tag_time_of_day(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate and tag time of day information from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with time-of-day-related tags
        """
        # Convert to HSV
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Extract V channel (brightness)
            v_channel = hsv[:, :, 2]
            
            # Calculate mean brightness
            mean_v = np.mean(v_channel)
            
            # Calculate ratio of bright pixels (V > 100)
            bright_pixels = np.sum(v_channel > 100)
            total_pixels = v_channel.size
            bright_ratio = bright_pixels / total_pixels
            
            # Calculate ratio of dark pixels (V < 50)
            dark_pixels = np.sum(v_channel < 50)
            dark_ratio = dark_pixels / total_pixels
            
            # Determine time of day
            if mean_v > 100 and bright_ratio > 0.5:
                time_of_day = 'day'
            elif mean_v < 50 or dark_ratio > 0.6:
                time_of_day = 'night'
            else:
                # Check for golden hour / sunset colors
                hsv_mean = np.mean(hsv, axis=(0, 1))
                if 15 <= hsv_mean[0] <= 30:  # Orange-yellow hue
                    time_of_day = 'sunset/sunrise'
                else:
                    time_of_day = 'unknown'
            
            # Create tags
            tags = {
                'time_of_day': time_of_day,
                'mean_brightness': float(mean_v),
                'bright_pixel_ratio': float(bright_ratio),
                'dark_pixel_ratio': float(dark_ratio)
            }
        else:
            # Grayscale image
            mean_gray = np.mean(image)
            
            # Determine time of day
            if mean_gray > 100:
                time_of_day = 'day'
            elif mean_gray < 50:
                time_of_day = 'night'
            else:
                time_of_day = 'unknown'
            
            # Create tags
            tags = {
                'time_of_day': time_of_day,
                'mean_brightness': float(mean_gray)
            }
        
        return tags
    
    def tag_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess and tag image quality information.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with quality-related tags
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute Laplacian variance for blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Compute standard deviation as a measure of contrast
        std_dev = np.std(gray)
        
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
            saturated_mask = np.logical_or(gray == 0, gray == 255)
        
        # Calculate ratio of saturated pixels
        saturation_ratio = np.sum(saturated_mask) / saturated_mask.size
        
        # Estimate noise level
        denoised = cv2.medianBlur(gray, 5)
        noise = cv2.absdiff(gray, denoised)
        noise_level = np.mean(noise)
        
        # Determine quality issues
        quality_issues = []
        if laplacian_var < 100:
            quality_issues.append('blurry')
        
        if std_dev < 40:
            quality_issues.append('low_contrast')
        
        if np.mean(gray) < 50:
            quality_issues.append('too_dark')
        elif np.mean(gray) > 200:
            quality_issues.append('too_bright')
        
        if saturation_ratio > 0.05:
            quality_issues.append('saturation')
        
        if noise_level > 5:
            quality_issues.append('noisy')
        
        # Create tags
        tags = {
            'laplacian_variance': float(laplacian_var),
            'is_blurry': bool(laplacian_var < 100),
            'contrast': float(std_dev),
            'has_low_contrast': bool(std_dev < 40),
            'saturation_ratio': float(saturation_ratio),
            'noise_level': float(noise_level),
            'quality_issues': quality_issues,
            'overall_quality': 'good' if not quality_issues else 'needs_improvement'
        }
        
        return tags
    
    def tag_image_size(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Tag image size information.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with size-related tags
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Determine orientation
        if aspect_ratio > 1.1:
            orientation = 'landscape'
        elif aspect_ratio < 0.9:
            orientation = 'portrait'
        else:
            orientation = 'square'
        
        # Create tags
        tags = {
            'width': width,
            'height': height,
            'aspect_ratio': float(aspect_ratio),
            'orientation': orientation,
            'resolution': width * height,
            'is_high_resolution': bool(width * height > 1000000)  # 1 megapixel
        }
        
        return tags
    
    def tag_all(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Apply all tagging methods to an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with all tags
        """
        # Apply all tagging methods
        face_tags = self.tag_faces(image)
        color_tags = self.tag_colors(image)
        feature_tags = self.tag_features(image)
        time_tags = self.tag_time_of_day(image)
        quality_tags = self.tag_image_quality(image)
        size_tags = self.tag_image_size(image)
        
        # Combine all tags
        all_tags = {
            'faces': face_tags,
            'colors': color_tags,
            'features': feature_tags,
            'time_of_day': time_tags,
            'quality': quality_tags,
            'size': size_tags,
            'tagged_at': datetime.now().isoformat()
        }
        
        return all_tags


# Example usage:
if __name__ == "__main__":
    # Initialize dataset creator
    creator = DatasetCreator("../data/datasets")
    
    # Create a new dataset
    dataset_id = creator.create_dataset("Dubai Landmarks", "landmarks", "Dataset of landmarks in Dubai")
    
    # Add classes
    creator.add_class(dataset_id, "burj_khalifa", "Burj Khalifa")
    creator.add_class(dataset_id, "dubai_marina", "Dubai Marina")
    creator.add_class(dataset_id, "palm_jumeirah", "Palm Jumeirah")
    
    # Add sample images if available
    sample_dir = "../data/images/landmarks"  # Update with your image directory
    if os.path.exists(sample_dir):
        # Add images from directory
        added_ids = creator.add_images_from_directory(dataset_id, sample_dir, True)
        print(f"Added {len(added_ids)} images to dataset")
    
    # Initialize image tagger
    tagger = ImageTagger()
    
    # Test tagging on a sample image
    sample_image_path = "../data/images/test.jpg"  # Update with your test image path
    if os.path.exists(sample_image_path):
        # Load image
        image = cv2.imread(sample_image_path)
        
        if image is not None:
            # Tag image
            tags = tagger.tag_all(image)
            
            # Print tags
            print("Image tags:")
            for category, category_tags in tags.items():
                print(f"  {category}:")
                for tag, value in category_tags.items():
                    if isinstance(value, list) and len(value) > 10:
                        print(f"    {tag}: [List with {len(value)} items]")
                    else:
                        print(f"    {tag}: {value}")
            
            # Add tagged image to dataset
            if len(creator.list_datasets()) > 0:
                dataset_id = creator.list_datasets()[0]['id']
                
                # Determine class from tags
                if tags['faces']['has_faces']:
                    class_name = "people"  # Example class
                else:
                    class_name = "other"  # Example class
                
                # Add class if it doesn't exist
                if class_name not in creator.get_dataset_info(dataset_id)['classes']:
                    creator.add_class(dataset_id, class_name)
                
                # Add image with tags
                image_id = creator.add_image(dataset_id, sample_image_path, class_name)
                
                # Add tags to image
                creator.tag_image(dataset_id, image_id, {
                    'automatic_tags': tags,
                    'original_filename': os.path.basename(sample_image_path)
                })
                
                print(f"Added and tagged image with ID: {image_id}")
        else:
            print(f"Could not load image: {sample_image_path}")
    else:
        print(f"Sample image not found: {sample_image_path}")
