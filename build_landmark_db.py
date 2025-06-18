import os
import cv2
import sys
from src.modules.landmark_recognition import LandmarkRecognizer

def build_landmark_database_from_directory(directory, landmark_name, output_path="data/models/landmark_database"):
    """
    Build a landmark database from a directory of images.
    
    Args:
        directory: Directory containing landmark images
        landmark_name: Name of the landmark
        output_path: Path to save the landmark database
    """
    # Initialize the landmark recognizer
    recognizer = LandmarkRecognizer()
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return False
    
    # Process each image in the directory
    image_count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            
            # Load image
            image = cv2.imread(image_path)
            
            if image is not None:
                # Add to landmark database
                recognizer.add_landmark(landmark_name, image)
                image_count += 1
                print(f"Added image: {filename}")
            else:
                print(f"Could not load image: {image_path}")
    
    print(f"Added {image_count} images for landmark '{landmark_name}'")
    
    if image_count == 0:
        print("No images were added")
        return False
    
    # Save the landmark database
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recognizer.save_database(output_path)
    print(f"Landmark database saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_landmark_db_from_dir.py <directory> <landmark_name> [output_path]")
        sys.exit(1)
    
    directory = sys.argv[1]
    landmark_name = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/models/landmark_database"
    
    success = build_landmark_database_from_directory(directory, landmark_name, output_path)
    sys.exit(0 if success else 1)
