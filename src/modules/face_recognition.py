import cv2
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


import cv2
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class FaceDetector:
    """
    Class for detecting faces in images using OpenCV's DNN-based SSD face detector.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the DNN face detector using paths relative to this file.
        """
        base_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
        prototxt_path = os.path.join(base_path, 'deploy.prototxt')
        model_path = os.path.join(base_path, 'res10_300x300_ssd_iter_140000.caffemodel')

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold


    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image using a DNN model.

        Args:
            image: Input image (BGR format)

        Returns:
            List of face bounding boxes in format (x, y, width, height)
        """
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.

        Args:
            image: Input image
            faces: List of face bounding boxes in format (x, y, width, height)
            color: Color of the bounding box (BGR format)
            thickness: Thickness of the bounding box lines

        Returns:
            Image with face bounding boxes drawn
        """
        img_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        return img_copy

    def extract_face_regions(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                             target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Extract face regions from the image.

        Args:
            image: Input image
            faces: List of face bounding boxes in format (x, y, width, height)
            target_size: Size to resize extracted faces to (optional)

        Returns:
            List of extracted face images
        """
        face_regions = []
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            if target_size is not None:
                face = cv2.resize(face, target_size)
            face_regions.append(face)
        return face_regions

    
    
class FaceRecognizer:
    """
    Class for recognizing faces using OpenCV's LBPH Face Recognizer.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the face recognizer.
        
        Args:
            model_path: Path to a pre-trained face recognition model (optional)
        """
        # Create LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load pre-trained model if provided
        if model_path is not None and os.path.exists(model_path):
            self.recognizer.read(model_path)
            self.is_trained = True
        else:
            self.is_trained = False
            
        # Dictionary to map label IDs to names
        self.label_names = {}
    
    def train(self, faces: List[np.ndarray], labels: List[int], label_names: Optional[Dict[int, str]] = None) -> None:
        """
        Train the face recognizer with face images and corresponding labels.
        
        Args:
            faces: List of face images (grayscale)
            labels: List of integer labels corresponding to each face
            label_names: Dictionary mapping label IDs to names (optional)
        """
        # Convert faces to grayscale if they aren't already
        gray_faces = []
        for face in faces:
            if len(face.shape) == 3:  # Color image
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_faces.append(gray_face)
            else:  # Already grayscale
                gray_faces.append(face)
        
        # Train the recognizer
        self.recognizer.train(gray_faces, np.array(labels))
        self.is_trained = True
        
        # Store label names if provided
        if label_names is not None:
            self.label_names = label_names
    
    def predict(self, face: np.ndarray) -> Tuple[int, float, Optional[str]]:
        """
        Predict the identity of a face.
        
        Args:
            face: Face image
            
        Returns:
            Tuple of (label, confidence, name) where name is the label name if available
        """
        if not self.is_trained:
            raise ValueError("Face recognizer has not been trained yet")
        
        # Convert to grayscale if needed
        if len(face.shape) == 3:  # Color image
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            gray_face = face
        
        # Predict
        label, confidence = self.recognizer.predict(gray_face)
        
        # Get name if available
        name = self.label_names.get(label)
        
        return label, confidence, name
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained face recognition model.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save model: Face recognizer has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.recognizer.write(model_path)
    
    def load_model(self, model_path: str, label_names: Optional[Dict[int, str]] = None) -> None:
        """
        Load a pre-trained face recognition model.
        
        Args:
            model_path: Path to the model file
            label_names: Dictionary mapping label IDs to names (optional)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        self.recognizer.read(model_path)
        self.is_trained = True
        
        # Store label names if provided
        if label_names is not None:
            self.label_names = label_names


def create_face_dataset(detector: FaceDetector, image_dir: str, output_dir: str, 
                       target_size: Tuple[int, int] = (100, 100)) -> Dict[str, List[str]]:
    """
    Create a face dataset from a directory of images.
    
    Args:
        detector: FaceDetector instance
        image_dir: Directory containing images
        output_dir: Directory to save extracted faces
        target_size: Size to resize extracted faces to
        
    Returns:
        Dictionary mapping person names to lists of face image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store face image paths for each person
    face_dataset = {}
    
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Parse person name from filename (assuming format: person_name_XXX.jpg)
            parts = filename.split('_')
            if len(parts) >= 2:
                person_name = parts[0]
                
                # Create person directory if it doesn't exist
                person_dir = os.path.join(output_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)
                
                # Load image
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Detect faces
                    faces = detector.detect_faces(image)
                    
                    # Extract and save face regions
                    face_regions = detector.extract_face_regions(image, faces, target_size)
                    
                    for i, face in enumerate(face_regions):
                        # Generate output filename
                        output_filename = f"{person_name}_{os.path.splitext(filename)[0]}_{i}.jpg"
                        output_path = os.path.join(person_dir, output_filename)
                        
                        # Save face image
                        cv2.imwrite(output_path, face)
                        
                        # Add to dataset dictionary
                        if person_name not in face_dataset:
                            face_dataset[person_name] = []
                        face_dataset[person_name].append(output_path)
    
    return face_dataset


def load_face_dataset(dataset_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """
    Load a face dataset from a directory structure.
    
    Args:
        dataset_dir: Directory containing subdirectories for each person
        
    Returns:
        Tuple of (faces, labels, label_names) where:
            faces: List of face images
            labels: List of integer labels
            label_names: Dictionary mapping label IDs to names
    """
    faces = []
    labels = []
    label_names = {}
    label_id = 0
    
    # Process each person's directory
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        
        if os.path.isdir(person_dir):
            # Assign a label ID to this person
            label_names[label_id] = person_name
            
            # Process each face image
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    
                    # Load face image
                    face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if face is not None:
                        # Add to dataset
                        faces.append(face)
                        labels.append(label_id)
            
            # Increment label ID for next person
            label_id += 1
    
    return faces, labels, label_names


# Example usage:
if __name__ == "__main__":
    detector = FaceDetector()
    image_path = "../data/images/sample.jpg"

    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        faces = detector.detect_faces(image)
        result = detector.draw_faces(image, faces)

        cv2.imshow("Frontal + Profile Face Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Detected {len(faces)} faces (frontal and profile combined)")
    else:
        print(f"Image not found: {image_path}")

