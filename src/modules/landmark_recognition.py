import cv2
import os
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Any


class LandmarkRecognizer:
    """
    Class for recognizing landmarks in images using ORB features and feature matching.
    """
    
    def __init__(self, max_features: int = 1000, orb_distance_thresh: int = 65, 
                 min_matches: int = 8, ransac_thresh: float = 6.4):
        """
        Initialize the landmark recognizer.
        
        Args:
            max_features: Maximum number of features to detect
            orb_distance_thresh: Distance threshold for ORB matching
            min_matches: Minimum number of matches required for recognition
            ransac_thresh: RANSAC threshold for homography filtering
        """
        # Initialize ORB detector with enhanced parameters
        self.orb = cv2.ORB_create(nfeatures=max_features)
        
        # Initialize BFMatcher with Hamming distance (appropriate for ORB binary descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Enhanced parameters from the second script
        self.orb_distance_thresh = orb_distance_thresh
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        
        # Database of landmarks
        self.landmark_db = {}  # Maps landmark names to (keypoints, descriptors) tuples
        self.landmark_images = {}  # Maps landmark names to original images
    
    def add_landmark(self, name: str, image: np.ndarray) -> None:
        """
        Add a landmark to the database.
        
        Args:
            name: Name of the landmark
            image: Image of the landmark
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is not None and len(keypoints) > 0:
            # Store in database
            self.landmark_db[name] = (keypoints, descriptors)
            self.landmark_images[name] = image.copy()
        else:
            print(f"Warning: No features detected for landmark '{name}'")
    
    def match_orb_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match ORB features using the enhanced method from the second script.
        
        Args:
            desc1: Query descriptors
            desc2: Database descriptors
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < self.orb_distance_thresh]
        return good_matches
    
    def filter_matches_ransac(self, kp1: List, kp2: List, matches: List) -> List:
        """
        Filter matches using RANSAC homography estimation.
        
        Args:
            kp1: Query keypoints
            kp2: Database keypoints
            matches: Raw matches
            
        Returns:
            Filtered inlier matches
        """
        if len(matches) < 4:
            return []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            
            if mask is not None:
                inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
                return inlier_matches
        except:
            pass
        
        return []
    
    def recognize_landmark(self, image: np.ndarray, use_ransac: bool = True) -> Tuple[Optional[str], int, Optional[np.ndarray]]:
        """
        Recognize a landmark in the input image using enhanced ORB matching.
        
        Args:
            image: Input image
            use_ransac: Whether to use RANSAC filtering for matches
            
        Returns:
            Tuple of (landmark_name, match_count, result_image) where:
                landmark_name: Name of the recognized landmark, or None if no match
                match_count: Number of good matches
                result_image: Image with matches drawn, or None if no match
        """
        if not self.landmark_db:
            raise ValueError("Landmark database is empty. Add landmarks using add_landmark() first.")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors for the query image
        query_keypoints, query_descriptors = self.orb.detectAndCompute(gray, None)
        
        if query_descriptors is None or len(query_keypoints) == 0:
            return None, 0, None
        
        # Find the best match among all landmarks
        best_match = None
        best_match_count = 0
        best_matches = None
        best_landmark_keypoints = None
        best_landmark_name = None
        
        for name, (landmark_keypoints, landmark_descriptors) in self.landmark_db.items():
            # Match descriptors using enhanced method
            raw_matches = self.match_orb_features(query_descriptors, landmark_descriptors)
            
            # Apply RANSAC filtering if requested
            if use_ransac:
                good_matches = self.filter_matches_ransac(query_keypoints, landmark_keypoints, raw_matches)
            else:
                good_matches = raw_matches
            
            # Update best match if this is better
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_matches = good_matches
                best_landmark_keypoints = landmark_keypoints
                best_landmark_name = name
        
        # Check if the best match has enough matches
        if best_match_count >= self.min_matches:
            # Draw matches for visualization
            landmark_image = self.landmark_images[best_landmark_name]
            result_image = cv2.drawMatches(image, query_keypoints, landmark_image, best_landmark_keypoints, 
                                         best_matches[:min(50, len(best_matches))], None, 
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return best_landmark_name, best_match_count, result_image
        else:
            return None, 0, None
    
    def get_match_confidence(self, image: np.ndarray, landmark_name: str) -> float:
        """
        Calculate matching confidence for a specific landmark.
        
        Args:
            image: Input image
            landmark_name: Name of the landmark to match against
            
        Returns:
            Confidence score (ratio of matches to query keypoints)
        """
        if landmark_name not in self.landmark_db:
            return 0.0
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors for the query image
        query_keypoints, query_descriptors = self.orb.detectAndCompute(gray, None)
        
        if query_descriptors is None or len(query_keypoints) == 0:
            return 0.0
        
        # Get landmark data
        landmark_keypoints, landmark_descriptors = self.landmark_db[landmark_name]
        
        # Match and filter
        raw_matches = self.match_orb_features(query_descriptors, landmark_descriptors)
        good_matches = self.filter_matches_ransac(query_keypoints, landmark_keypoints, raw_matches)
        
        # Calculate confidence
        confidence = len(good_matches) / len(query_keypoints) if query_keypoints else 0.0
        return confidence
    
    def save_database(self, db_dir: str) -> None:
        """
        Save the landmark database to disk.
        
        Args:
            db_dir: Directory to save the database
        """
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        # Save each landmark
        for name, (keypoints, descriptors) in self.landmark_db.items():
            # Create a safe filename
            safe_name = ''.join(c if c.isalnum() else '_' for c in name)
            
            # Save descriptors
            desc_path = os.path.join(db_dir, f"{safe_name}_descriptors.npy")
            np.save(desc_path, descriptors)
            
            # Save the image
            img_path = os.path.join(db_dir, f"{safe_name}_image.jpg")
            cv2.imwrite(img_path, self.landmark_images[name])
            
            # Save keypoint information using pickle (convert to serializable format)
            keypoints_data = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                             for kp in keypoints]
            kp_path = os.path.join(db_dir, f"{safe_name}_keypoints.pkl")
            with open(kp_path, 'wb') as f:
                pickle.dump(keypoints_data, f)
        
        # Save the landmark names
        names_path = os.path.join(db_dir, "landmark_names.txt")
        with open(names_path, 'w') as f:
            for name in self.landmark_db.keys():
                f.write(f"{name}\n")
    
    def load_database(self, db_dir: str) -> None:
        """
        Load the landmark database from disk.
        
        Args:
            db_dir: Directory containing the database
        """
        # Check if directory exists
        if not os.path.isdir(db_dir):
            raise FileNotFoundError(f"Database directory not found: {db_dir}")
        
        # Load landmark names
        names_path = os.path.join(db_dir, "landmark_names.txt")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Landmark names file not found: {names_path}")
        
        # Clear existing database
        self.landmark_db = {}
        self.landmark_images = {}
        
        # Load each landmark
        with open(names_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    # Create a safe filename
                    safe_name = ''.join(c if c.isalnum() else '_' for c in name)
                    
                    # Load descriptors
                    desc_path = os.path.join(db_dir, f"{safe_name}_descriptors.npy")
                    if os.path.exists(desc_path):
                        descriptors = np.load(desc_path)
                    else:
                        print(f"Warning: Descriptors not found for landmark '{name}'")
                        continue
                    
                    # Load image
                    img_path = os.path.join(db_dir, f"{safe_name}_image.jpg")
                    if os.path.exists(img_path):
                        image = cv2.imread(img_path)
                    else:
                        print(f"Warning: Image not found for landmark '{name}'")
                        continue
                    
                    # Load keypoints
                    kp_path = os.path.join(db_dir, f"{safe_name}_keypoints.pkl")
                    if os.path.exists(kp_path):
                        with open(kp_path, 'rb') as f:
                            keypoints_data = pickle.load(f)
                    else:
                        print(f"Warning: Keypoints not found for landmark '{name}'")
                        continue
                    
                    # Convert keypoints back to OpenCV KeyPoint objects
                    keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], 
                                            angle=pt[2], response=pt[3], octave=pt[4], 
                                            class_id=pt[5]) for pt in keypoints_data]
                    
                    # Add to database
                    self.landmark_db[name] = (keypoints, descriptors)
                    self.landmark_images[name] = image


def create_landmark_dataset(image_dir: str, output_dir: str) -> Dict[str, List[str]]:
    """
    Create a landmark dataset from a directory of images.
    
    Args:
        image_dir: Directory containing images organized in subdirectories by landmark name
        output_dir: Directory to save processed landmark images
        
    Returns:
        Dictionary mapping landmark names to lists of image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store image paths for each landmark
    landmark_dataset = {}
    
    # Process each landmark directory
    for landmark_name in os.listdir(image_dir):
        landmark_dir = os.path.join(image_dir, landmark_name)
        
        if os.path.isdir(landmark_dir):
            # Create landmark directory in output directory
            output_landmark_dir = os.path.join(output_dir, landmark_name)
            os.makedirs(output_landmark_dir, exist_ok=True)
            
            # Process each image
            for filename in os.listdir(landmark_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Load image
                    image_path = os.path.join(landmark_dir, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        # Generate output filename
                        output_filename = f"{landmark_name}_{filename}"
                        output_path = os.path.join(output_landmark_dir, output_filename)
                        
                        # Save processed image
                        cv2.imwrite(output_path, image)
                        
                        # Add to dataset dictionary
                        if landmark_name not in landmark_dataset:
                            landmark_dataset[landmark_name] = []
                        landmark_dataset[landmark_name].append(output_path)
    
    return landmark_dataset


def load_landmark_dataset(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Load a landmark dataset from a directory structure.
    
    Args:
        dataset_dir: Directory containing subdirectories for each landmark
        
    Returns:
        Dictionary mapping landmark names to lists of image paths
    """
    landmark_dataset = {}
    
    # Process each landmark directory
    for landmark_name in os.listdir(dataset_dir):
        landmark_dir = os.path.join(dataset_dir, landmark_name)
        
        if os.path.isdir(landmark_dir):
            # Process each image
            for filename in os.listdir(landmark_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(landmark_dir, filename)
                    
                    # Add to dataset dictionary
                    if landmark_name not in landmark_dataset:
                        landmark_dataset[landmark_name] = []
                    landmark_dataset[landmark_name].append(image_path)
    
    return landmark_dataset


# Example usage:
if __name__ == "__main__":
    # Initialize landmark recognizer with enhanced ORB parameters
    recognizer = LandmarkRecognizer(
        max_features=1000,
        orb_distance_thresh=65,
        min_matches=8,
        ransac_thresh=6.4
    )
    
    # Test on sample images
    sample_dir = "../data/images/landmarks"  # Update with your landmark image directory
    if os.path.exists(sample_dir):
        # Load landmark database
        for landmark_name in os.listdir(sample_dir):
            landmark_dir = os.path.join(sample_dir, landmark_name)
            
            if os.path.isdir(landmark_dir):
                # Use the first image as the reference
                for filename in os.listdir(landmark_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(landmark_dir, filename)
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Add to database
                            recognizer.add_landmark(landmark_name, image)
                            print(f"Added landmark: {landmark_name}")
                            break
        
        # Test recognition on a sample image
        test_image_path = "../data/images/test.jpg"  # Update with your test image path
        if os.path.exists(test_image_path):
            # Load image
            test_image = cv2.imread(test_image_path)
            
            # Recognize landmark
            landmark_name, match_count, result_image = recognizer.recognize_landmark(test_image)
            
            if landmark_name is not None:
                print(f"Recognized landmark: {landmark_name} with {match_count} matches")
                
                # Get confidence score
                confidence = recognizer.get_match_confidence(test_image, landmark_name)
                print(f"Confidence score: {confidence:.3f}")
                
                # Display result
                cv2.imshow("Landmark Recognition", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No landmark recognized")
        else:
            print(f"Test image not found: {test_image_path}")
    else:
        print(f"Sample directory not found: {sample_dir}")