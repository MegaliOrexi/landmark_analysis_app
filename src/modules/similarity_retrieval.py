import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


class SimilarityRetriever:
    """
    Class for retrieving similar images based on various similarity metrics.
    """
    
    def __init__(self, feature_type: str = 'orb'):
        """
        Initialize the similarity retriever.
        
        Args:
            feature_type: Type of features to use ('orb', 'sift', 'color_hist', 'hog', 'combined')
        """
        self.feature_type = feature_type
        
        # Initialize feature extractors based on type
        if feature_type == 'orb' or feature_type == 'combined':
            self.orb = cv2.ORB_create(nfeatures=1000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if feature_type == 'sift' or feature_type == 'combined':
            # Check if SIFT is available (OpenCV contrib)
            try:
                self.sift = cv2.SIFT_create()
                self.flann_matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
            except AttributeError:
                print("Warning: SIFT not available. Using ORB instead.")
                self.feature_type = 'orb' if feature_type == 'sift' else 'combined_no_sift'
                self.orb = cv2.ORB_create(nfeatures=1000)
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Database of images and their features
        self.image_paths = []
        self.image_features = []
        self.image_metadata = []
    
    def _extract_orb_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def _extract_sift_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Color histogram features
        """
        # Check if image is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        
        # Color image
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate histograms
        hist_features = np.concatenate((h_hist, s_hist, v_hist))
        
        return hist_features
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            HOG features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize image to a fixed size
        resized = cv2.resize(gray, (128, 128))
        
        # Calculate HOG features
        # Parameters: image, win_size, block_size, block_stride, cell_size, nbins
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(resized)
        
        return hog_features.flatten()
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from an image based on the configured feature type.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        if self.feature_type == 'orb' or self.feature_type == 'combined' or self.feature_type == 'combined_no_sift':
            keypoints, descriptors = self._extract_orb_features(image)
            features['orb_keypoints'] = keypoints
            features['orb_descriptors'] = descriptors
        
        if self.feature_type == 'sift' or self.feature_type == 'combined':
            keypoints, descriptors = self._extract_sift_features(image)
            features['sift_keypoints'] = keypoints
            features['sift_descriptors'] = descriptors
        
        if self.feature_type == 'color_hist' or self.feature_type == 'combined' or self.feature_type == 'combined_no_sift':
            hist_features = self._extract_color_histogram(image)
            features['color_hist'] = hist_features
        
        if self.feature_type == 'hog' or self.feature_type == 'combined' or self.feature_type == 'combined_no_sift':
            hog_features = self._extract_hog_features(image)
            features['hog'] = hog_features
        
        return features
    
    def add_image(self, image_path: str, image: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an image to the database.
        
        Args:
            image_path: Path to the image file
            image: Image data (optional, will be loaded from path if not provided)
            metadata: Additional metadata for the image (optional)
        """
        # Load image if not provided
        if image is None:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image from {image_path}")
                return
        
        # Extract features
        features = self.extract_features(image)
        
        # Add to database
        self.image_paths.append(image_path)
        self.image_features.append(features)
        self.image_metadata.append(metadata or {})
    
    def add_images_from_directory(self, directory: str, recursive: bool = False, 
                                extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> int:
        """
        Add all images from a directory to the database.
        
        Args:
            directory: Directory containing images
            recursive: Whether to search subdirectories recursively
            extensions: List of valid file extensions
            
        Returns:
            Number of images added
        """
        # Check if directory exists
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Count of added images
        added_count = 0
        
        # Function to process a directory
        def process_dir(dir_path, relative_path=''):
            nonlocal added_count
            
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                if os.path.isdir(file_path) and recursive:
                    # Process subdirectory
                    new_relative_path = os.path.join(relative_path, filename)
                    process_dir(file_path, new_relative_path)
                elif any(filename.lower().endswith(ext) for ext in extensions):
                    # Load image
                    image = cv2.imread(file_path)
                    
                    if image is not None:
                        # Create metadata
                        metadata = {
                            'filename': filename,
                            'directory': dir_path,
                            'relative_path': os.path.join(relative_path, filename)
                        }
                        
                        # Add image to database
                        self.add_image(file_path, image, metadata)
                        added_count += 1
                        
                        if added_count % 10 == 0:
                            print(f"Added {added_count} images...")
        
        # Process the root directory
        process_dir(directory)
        
        print(f"Added {added_count} images from {directory}")
        return added_count
    
    def _compute_orb_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute similarity between two images using ORB features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Tuple of (similarity_score, matches_image)
        """
        # Check if both images have ORB descriptors
        if 'orb_descriptors' not in query_features or query_features['orb_descriptors'] is None:
            return 0.0, None
        
        if 'orb_descriptors' not in db_features or db_features['orb_descriptors'] is None:
            return 0.0, None
        
        # Match descriptors
        matches = self.bf_matcher.match(query_features['orb_descriptors'], db_features['orb_descriptors'])
        
        # Sort matches by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate similarity score (number of good matches)
        # Use a threshold based on the minimum distance
        if len(matches) > 0:
            min_distance = matches[0].distance
            good_matches = [m for m in matches if m.distance < 3 * min_distance]
            similarity_score = len(good_matches)
        else:
            similarity_score = 0
            good_matches = []
        
        return similarity_score, good_matches
    
    def _compute_sift_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute similarity between two images using SIFT features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Tuple of (similarity_score, matches_image)
        """
        # Check if both images have SIFT descriptors
        if 'sift_descriptors' not in query_features or query_features['sift_descriptors'] is None:
            return 0.0, None
        
        if 'sift_descriptors' not in db_features or db_features['sift_descriptors'] is None:
            return 0.0, None
        
        # Match descriptors
        matches = self.flann_matcher.knnMatch(query_features['sift_descriptors'], db_features['sift_descriptors'], k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Calculate similarity score (number of good matches)
        similarity_score = len(good_matches)
        
        return similarity_score, good_matches
    
    def _compute_color_hist_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> float:
        """
        Compute similarity between two images using color histogram features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Similarity score
        """
        # Check if both images have color histogram features
        if 'color_hist' not in query_features or 'color_hist' not in db_features:
            return 0.0
        
        # Calculate correlation between histograms
        correlation = cv2.compareHist(
            np.float32(query_features['color_hist']).reshape(-1, 1),
            np.float32(db_features['color_hist']).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        # Convert correlation to a similarity score (0-100)
        similarity_score = max(0, correlation * 100)
        
        return similarity_score
    
    def _compute_hog_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> float:
        """
        Compute similarity between two images using HOG features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Similarity score
        """
        # Check if both images have HOG features
        if 'hog' not in query_features or 'hog' not in db_features:
            return 0.0
        
        # Calculate cosine similarity between HOG features
        query_hog = query_features['hog']
        db_hog = db_features['hog']
        
        # Normalize features
        query_norm = np.linalg.norm(query_hog)
        db_norm = np.linalg.norm(db_hog)
        
        if query_norm == 0 or db_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(query_hog, db_hog) / (query_norm * db_norm)
        
        # Convert to a similarity score (0-100)
        similarity_score = max(0, cosine_similarity * 100)
        
        return similarity_score
    
    def compute_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute similarity between two images based on the configured feature type.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Tuple of (similarity_score, similarity_details)
        """
        similarity_details = {}
        
        if self.feature_type == 'orb':
            orb_score, orb_matches = self._compute_orb_similarity(query_features, db_features)
            similarity_details['orb'] = {'score': orb_score, 'matches': orb_matches}
            return orb_score, similarity_details
        
        elif self.feature_type == 'sift':
            sift_score, sift_matches = self._compute_sift_similarity(query_features, db_features)
            similarity_details['sift'] = {'score': sift_score, 'matches': sift_matches}
            return sift_score, similarity_details
        
        elif self.feature_type == 'color_hist':
            color_score = self._compute_color_hist_similarity(query_features, db_features)
            similarity_details['color_hist'] = {'score': color_score}
            return color_score, similarity_details
        
        elif self.feature_type == 'hog':
            hog_score = self._compute_hog_similarity(query_features, db_features)
            similarity_details['hog'] = {'score': hog_score}
            return hog_score, similarity_details
        
        elif self.feature_type == 'combined' or self.feature_type == 'combined_no_sift':
            # Compute scores for each feature type
            scores = {}
            
            # ORB features
            orb_score, orb_matches = self._compute_orb_similarity(query_features, db_features)
            scores['orb'] = orb_score
            similarity_details['orb'] = {'score': orb_score, 'matches': orb_matches}
            
            # SIFT features (if available)
            if self.feature_type == 'combined':
                sift_score, sift_matches = self._compute_sift_similarity(query_features, db_features)
                scores['sift'] = sift_score
                similarity_details['sift'] = {'score': sift_score, 'matches': sift_matches}
            
            # Color histogram features
            color_score = self._compute_color_hist_similarity(query_features, db_features)
            scores['color_hist'] = color_score
            similarity_details['color_hist'] = {'score': color_score}
            
            # HOG features
            hog_score = self._compute_hog_similarity(query_features, db_features)
            scores['hog'] = hog_score
            similarity_details['hog'] = {'score': hog_score}
            
            # Normalize scores
            max_scores = {
                'orb': 100,  # Assuming 100 is a good number of matches
                'sift': 100,  # Assuming 100 is a good number of matches
                'color_hist': 100,  # Already normalized to 0-100
                'hog': 100  # Already normalized to 0-100
            }
            
            normalized_scores = {}
            for feature, score in scores.items():
                normalized_scores[feature] = min(1.0, score / max_scores[feature])
            
            # Assign weights to each feature type
            weights = {
                'orb': 0.4,
                'sift': 0.3,
                'color_hist': 0.2,
                'hog': 0.1
            }
            
            # If SIFT is not available, redistribute its weight
            if self.feature_type == 'combined_no_sift':
                weights['orb'] += weights['sift'] * 0.6
                weights['color_hist'] += weights['sift'] * 0.3
                weights['hog'] += weights['sift'] * 0.1
                weights['sift'] = 0
            
            # Calculate weighted average
            weighted_score = 0
            for feature, score in normalized_scores.items():
                weighted_score += score * weights[feature]
            
            # Convert to a similarity score (0-100)
            similarity_score = weighted_score * 100
            
            # Add combined score to details
            similarity_details['combined'] = {
                'score': similarity_score,
                'normalized_scores': normalized_scores,
                'weights': weights
            }
            
            return similarity_score, similarity_details
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def retrieve_similar_images(self, query_image: np.ndarray, top_k: int = 3, 
                              similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar images to a query image.
        
        Args:
            query_image: Query image (BGR format)
            top_k: Number of similar images to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with similar image information
        """
        # Check if database is empty
        if not self.image_features:
            print("Warning: Image database is empty")
            return []
        
        # Extract features from query image
        query_features = self.extract_features(query_image)
        
        # Compute similarity with each image in the database
        similarities = []
        
        for i, db_features in enumerate(self.image_features):
            similarity_score, similarity_details = self.compute_similarity(query_features, db_features)
            
            if similarity_score >= similarity_threshold:
                similarities.append({
                    'index': i,
                    'path': self.image_paths[i],
                    'metadata': self.image_metadata[i],
                    'similarity_score': similarity_score,
                    'similarity_details': similarity_details
                })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
    
    def visualize_similar_images(self, query_image: np.ndarray, similar_images: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize query image and its similar images.
        
        Args:
            query_image: Query image (BGR format)
            similar_images: List of similar image dictionaries from retrieve_similar_images()
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        # Number of similar images
        n_similar = len(similar_images)
        
        if n_similar == 0:
            print("No similar images found")
            return query_image
        
        # Create figure
        fig, axs = plt.subplots(1, n_similar + 1, figsize=(4 * (n_similar + 1), 4))
        
        # Display query image
        axs[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Query Image")
        axs[0].axis('off')
        
        # Display similar images
        for i, similar in enumerate(similar_images):
            # Load similar image
            similar_image = cv2.imread(similar['path'])
            
            if similar_image is not None:
                # Display image
                axs[i + 1].imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
                axs[i + 1].set_title(f"Similarity: {similar['similarity_score']:.2f}")
                axs[i + 1].axis('off')
                
                # If using ORB or SIFT, draw matches
                if self.feature_type in ['orb', 'combined', 'combined_no_sift'] and 'orb' in similar['similarity_details']:
                    # Get matches
                    orb_details = similar['similarity_details']['orb']
                    if 'matches' in orb_details and orb_details['matches']:
                        # Draw matches on a separate subplot
                        fig.set_figheight(8)
                        if len(axs.shape) == 1:
                            axs = axs.reshape(1, -1)
                        
                        if i == 0:
                            # Add a new row of subplots for matches
                            fig.delaxes(axs[0, 0])
                            axs = np.vstack((axs, np.zeros((1, n_similar + 1), dtype=object)))
                            axs[0, 0] = fig.add_subplot(2, n_similar + 1, 1)
                            axs[0, 0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
                            axs[0, 0].set_title("Query Image")
                            axs[0, 0].axis('off')
                            
                            for j in range(1, n_similar + 1):
                                axs[0, j] = fig.add_subplot(2, n_similar + 1, j + 1)
                        
                        # Draw matches
                        query_keypoints = query_features['orb_keypoints']
                        db_features = self.image_features[similar['index']]
                        db_keypoints = db_features['orb_keypoints']
                        
                        matches_img = cv2.drawMatches(
                            query_image, query_keypoints,
                            similar_image, db_keypoints,
                            orb_details['matches'][:10],  # Show top 10 matches
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        
                        axs[1, i + 1] = fig.add_subplot(2, n_similar + 1, n_similar + 2 + i + 1)
                        axs[1, i + 1].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
                        axs[1, i + 1].set_title(f"Top Matches: {len(orb_details['matches'])}")
                        axs[1, i + 1].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path is not None:
            plt.savefig(output_path)
        
        # Convert figure to image
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return vis_image
    
    def save_database(self, db_path: str) -> None:
        """
        Save the image database to disk.
        
        Args:
            db_path: Path to save the database
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create database dictionary
        database = {
            'feature_type': self.feature_type,
            'image_paths': self.image_paths,
            'image_features': self.image_features,
            'image_metadata': self.image_metadata
        }
        
        # Save to file
        with open(db_path, 'wb') as f:
            pickle.dump(database, f)
    
    def load_database(self, db_path: str) -> None:
        """
        Load the image database from disk.
        
        Args:
            db_path: Path to the database file
        """
        # Check if file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Load from file
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        
        # Check if database format is valid
        required_keys = ['feature_type', 'image_paths', 'image_features', 'image_metadata']
        for key in required_keys:
            if key not in database:
                raise ValueError(f"Invalid database format: Missing key '{key}'")
        
        # Update instance variables
        self.feature_type = database['feature_type']
        self.image_paths = database['image_paths']
        self.image_features = database['image_features']
        self.image_metadata = database['image_metadata']
        
        # Initialize feature extractors based on feature type
        if self.feature_type == 'orb' or self.feature_type == 'combined' or self.feature_type == 'combined_no_sift':
            self.orb = cv2.ORB_create(nfeatures=1000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if self.feature_type == 'sift' or self.feature_type == 'combined':
            # Check if SIFT is available (OpenCV contrib)
            try:
                self.sift = cv2.SIFT_create()
                self.flann_matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
            except AttributeError:
                print("Warning: SIFT not available. Using ORB instead.")
                if self.feature_type == 'sift':
                    self.feature_type = 'orb'
                    self.orb = cv2.ORB_create(nfeatures=1000)
                    self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class EdgeSimilarityRetriever(SimilarityRetriever):
    """
    Class for retrieving similar images based on edge features.
    """
    
    def __init__(self):
        """
        Initialize the edge similarity retriever.
        """
        super().__init__(feature_type='orb')  # Use ORB as base
        
        # Override feature type
        self.feature_type = 'edge'
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract edge features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of edge features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge histogram
        hist = cv2.calcHist([edges], [0], None, [2], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Calculate edge direction using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and angle
        magnitude = cv2.magnitude(sobelx, sobely)
        angle = cv2.phase(sobelx, sobely, angleInDegrees=True)
        
        # Calculate angle histogram
        angle_hist = cv2.calcHist([angle.astype(np.float32)], [0], None, [36], [0, 360])
        angle_hist = cv2.normalize(angle_hist, angle_hist).flatten()
        
        # Return features
        return {
            'edges': edges,
            'edge_hist': hist,
            'magnitude': magnitude,
            'angle': angle,
            'angle_hist': angle_hist
        }
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of features
        """
        # Extract ORB features (for visualization)
        orb_features = super().extract_features(image)
        
        # Extract edge features
        edge_features = self._extract_edge_features(image)
        
        # Combine features
        features = {**orb_features, **edge_features}
        
        return features
    
    def compute_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute similarity between two images based on edge features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Tuple of (similarity_score, similarity_details)
        """
        similarity_details = {}
        
        # Check if both images have edge features
        if 'edge_hist' not in query_features or 'edge_hist' not in db_features:
            return 0.0, similarity_details
        
        # Calculate correlation between edge histograms
        edge_correlation = cv2.compareHist(
            np.float32(query_features['edge_hist']).reshape(-1, 1),
            np.float32(db_features['edge_hist']).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        # Calculate correlation between angle histograms
        angle_correlation = cv2.compareHist(
            np.float32(query_features['angle_hist']).reshape(-1, 1),
            np.float32(db_features['angle_hist']).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        # Calculate overall similarity score
        edge_weight = 0.4
        angle_weight = 0.6
        
        similarity_score = (edge_correlation * edge_weight + angle_correlation * angle_weight) * 100
        
        # Add details
        similarity_details['edge'] = {
            'score': similarity_score,
            'edge_correlation': edge_correlation,
            'angle_correlation': angle_correlation
        }
        
        # Also compute ORB similarity for visualization
        orb_score, orb_matches = self._compute_orb_similarity(query_features, db_features)
        similarity_details['orb'] = {'score': orb_score, 'matches': orb_matches}
        
        return similarity_score, similarity_details
    
    def visualize_similar_images(self, query_image: np.ndarray, similar_images: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize query image and its similar images with edge features.
        
        Args:
            query_image: Query image (BGR format)
            similar_images: List of similar image dictionaries from retrieve_similar_images()
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        # Number of similar images
        n_similar = len(similar_images)
        
        if n_similar == 0:
            print("No similar images found")
            return query_image
        
        # Create figure
        fig, axs = plt.subplots(3, n_similar + 1, figsize=(4 * (n_similar + 1), 12))
        
        # Extract edge features from query image
        query_features = self.extract_features(query_image)
        
        # Display query image
        axs[0, 0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Query Image")
        axs[0, 0].axis('off')
        
        # Display query image edges
        axs[1, 0].imshow(query_features['edges'], cmap='gray')
        axs[1, 0].set_title("Query Edges")
        axs[1, 0].axis('off')
        
        # Display query image angle
        axs[2, 0].imshow(query_features['angle'], cmap='hsv')
        axs[2, 0].set_title("Query Edge Direction")
        axs[2, 0].axis('off')
        
        # Display similar images
        for i, similar in enumerate(similar_images):
            # Load similar image
            similar_image = cv2.imread(similar['path'])
            
            if similar_image is not None:
                # Extract edge features
                db_features = self.image_features[similar['index']]
                
                # Display image
                axs[0, i + 1].imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
                axs[0, i + 1].set_title(f"Similarity: {similar['similarity_score']:.2f}")
                axs[0, i + 1].axis('off')
                
                # Display edges
                axs[1, i + 1].imshow(db_features['edges'], cmap='gray')
                axs[1, i + 1].set_title("Edges")
                axs[1, i + 1].axis('off')
                
                # Display angle
                axs[2, i + 1].imshow(db_features['angle'], cmap='hsv')
                axs[2, i + 1].set_title("Edge Direction")
                axs[2, i + 1].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path is not None:
            plt.savefig(output_path)
        
        # Convert figure to image
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return vis_image


class CornerSimilarityRetriever(SimilarityRetriever):
    """
    Class for retrieving similar images based on corner features.
    """
    
    def __init__(self):
        """
        Initialize the corner similarity retriever.
        """
        super().__init__(feature_type='orb')  # Use ORB as base
        
        # Override feature type
        self.feature_type = 'corner'
    
    def _extract_corner_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract corner features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of corner features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect corners using Harris corner detector
        harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        
        # Normalize and threshold
        harris_normalized = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, harris_threshold = cv2.threshold(harris_normalized, 20, 255, cv2.THRESH_BINARY)
        
        # Count corners
        harris_corner_count = np.sum(harris_threshold > 0)
        
        # Detect corners using Shi-Tomasi corner detector
        shi_tomasi_corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        # Count corners
        shi_tomasi_corner_count = 0 if shi_tomasi_corners is None else len(shi_tomasi_corners)
        
        # Create corner density map
        corner_density = np.zeros_like(gray)
        if shi_tomasi_corners is not None:
            for corner in shi_tomasi_corners:
                x, y = corner.ravel()
                cv2.circle(corner_density, (int(x), int(y)), 5, 255, -1)
        
        # Calculate corner density histogram
        density_hist = cv2.calcHist([corner_density], [0], None, [10], [0, 256])
        density_hist = cv2.normalize(density_hist, density_hist).flatten()
        
        # Calculate spatial distribution of corners
        corner_distribution = np.zeros(9)  # 3x3 grid
        if shi_tomasi_corners is not None:
            height, width = gray.shape
            for corner in shi_tomasi_corners:
                x, y = corner.ravel()
                grid_x = min(2, int(x / width * 3))
                grid_y = min(2, int(y / height * 3))
                corner_distribution[grid_y * 3 + grid_x] += 1
            
            # Normalize
            if np.sum(corner_distribution) > 0:
                corner_distribution = corner_distribution / np.sum(corner_distribution)
        
        # Return features
        return {
            'harris_corners': harris_normalized,
            'harris_corner_count': harris_corner_count,
            'shi_tomasi_corners': shi_tomasi_corners,
            'shi_tomasi_corner_count': shi_tomasi_corner_count,
            'corner_density': corner_density,
            'density_hist': density_hist,
            'corner_distribution': corner_distribution
        }
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of features
        """
        # Extract ORB features (for visualization)
        orb_features = super().extract_features(image)
        
        # Extract corner features
        corner_features = self._extract_corner_features(image)
        
        # Combine features
        features = {**orb_features, **corner_features}
        
        return features
    
    def compute_similarity(self, query_features: Dict[str, Any], db_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute similarity between two images based on corner features.
        
        Args:
            query_features: Features of the query image
            db_features: Features of the database image
            
        Returns:
            Tuple of (similarity_score, similarity_details)
        """
        similarity_details = {}
        
        # Check if both images have corner features
        if 'corner_distribution' not in query_features or 'corner_distribution' not in db_features:
            return 0.0, similarity_details
        
        # Calculate correlation between density histograms
        density_correlation = cv2.compareHist(
            np.float32(query_features['density_hist']).reshape(-1, 1),
            np.float32(db_features['density_hist']).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        # Calculate similarity between corner distributions
        query_dist = query_features['corner_distribution']
        db_dist = db_features['corner_distribution']
        
        # Calculate cosine similarity
        dist_similarity = np.dot(query_dist, db_dist) / (np.linalg.norm(query_dist) * np.linalg.norm(db_dist)) if np.linalg.norm(query_dist) * np.linalg.norm(db_dist) > 0 else 0
        
        # Calculate similarity based on corner counts
        query_count = query_features['shi_tomasi_corner_count']
        db_count = db_features['shi_tomasi_corner_count']
        
        count_ratio = min(query_count, db_count) / max(query_count, db_count) if max(query_count, db_count) > 0 else 0
        
        # Calculate overall similarity score
        density_weight = 0.3
        dist_weight = 0.5
        count_weight = 0.2
        
        similarity_score = (density_correlation * density_weight + dist_similarity * dist_weight + count_ratio * count_weight) * 100
        
        # Add details
        similarity_details['corner'] = {
            'score': similarity_score,
            'density_correlation': density_correlation,
            'distribution_similarity': dist_similarity,
            'count_ratio': count_ratio
        }
        
        # Also compute ORB similarity for visualization
        orb_score, orb_matches = self._compute_orb_similarity(query_features, db_features)
        similarity_details['orb'] = {'score': orb_score, 'matches': orb_matches}
        
        return similarity_score, similarity_details
    
    def visualize_similar_images(self, query_image: np.ndarray, similar_images: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize query image and its similar images with corner features.
        
        Args:
            query_image: Query image (BGR format)
            similar_images: List of similar image dictionaries from retrieve_similar_images()
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        # Number of similar images
        n_similar = len(similar_images)
        
        if n_similar == 0:
            print("No similar images found")
            return query_image
        
        # Create figure
        fig, axs = plt.subplots(2, n_similar + 1, figsize=(4 * (n_similar + 1), 8))
        
        # Extract corner features from query image
        query_features = self.extract_features(query_image)
        
        # Display query image
        axs[0, 0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Query Image")
        axs[0, 0].axis('off')
        
        # Display query image corners
        corner_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB).copy()
        if query_features['shi_tomasi_corners'] is not None:
            for corner in query_features['shi_tomasi_corners']:
                x, y = corner.ravel()
                cv2.circle(corner_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        axs[1, 0].imshow(corner_img)
        axs[1, 0].set_title(f"Corners: {query_features['shi_tomasi_corner_count']}")
        axs[1, 0].axis('off')
        
        # Display similar images
        for i, similar in enumerate(similar_images):
            # Load similar image
            similar_image = cv2.imread(similar['path'])
            
            if similar_image is not None:
                # Extract corner features
                db_features = self.image_features[similar['index']]
                
                # Display image
                axs[0, i + 1].imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
                axs[0, i + 1].set_title(f"Similarity: {similar['similarity_score']:.2f}")
                axs[0, i + 1].axis('off')
                
                # Display corners
                corner_img = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB).copy()
                if db_features['shi_tomasi_corners'] is not None:
                    for corner in db_features['shi_tomasi_corners']:
                        x, y = corner.ravel()
                        cv2.circle(corner_img, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                axs[1, i + 1].imshow(corner_img)
                axs[1, i + 1].set_title(f"Corners: {db_features['shi_tomasi_corner_count']}")
                axs[1, i + 1].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path is not None:
            plt.savefig(output_path)
        
        # Convert figure to image
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return vis_image


# Example usage:
if __name__ == "__main__":
    # Initialize similarity retriever
    retriever = SimilarityRetriever(feature_type='combined')
    
    # Test on sample images
    sample_dir = "../data/images/landmarks"  # Update with your image directory
    if os.path.exists(sample_dir):
        # Add images to database
        retriever.add_images_from_directory(sample_dir, recursive=True)
        
        # Test retrieval on a sample image
        test_image_path = "../data/images/test.jpg"  # Update with your test image path
        if os.path.exists(test_image_path):
            # Load image
            test_image = cv2.imread(test_image_path)
            
            if test_image is not None:
                # Retrieve similar images
                similar_images = retriever.retrieve_similar_images(test_image, top_k=3)
                
                # Print results
                print(f"Found {len(similar_images)} similar images:")
                for i, similar in enumerate(similar_images):
                    print(f"  {i+1}. {similar['path']} (Similarity: {similar['similarity_score']:.2f})")
                
                # Visualize results
                vis_image = retriever.visualize_similar_images(test_image, similar_images)
                
                # Display visualization
                cv2.imshow("Similar Images", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Could not load test image: {test_image_path}")
        else:
            print(f"Test image not found: {test_image_path}")
    else:
        print(f"Sample directory not found: {sample_dir}")
    
    # Test edge-based similarity retriever
    edge_retriever = EdgeSimilarityRetriever()
    
    # Test corner-based similarity retriever
    corner_retriever = CornerSimilarityRetriever()
