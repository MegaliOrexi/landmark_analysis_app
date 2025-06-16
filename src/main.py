#!/usr/bin/env python3
"""
Visual Landmark and Scene Analysis Application

This application provides functionality for landmark and face recognition,
time-of-day classification, image quality assessment, dataset creation,
similarity retrieval, and image annotation.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.modules.face_recognition import FaceDetector, FaceRecognizer
from src.modules.landmark_recognition import LandmarkRecognizer
from src.modules.time_of_day_classification import TimeOfDayClassifier
from src.modules.image_quality_assessment import ImageQualityAssessor
from src.modules.dataset_creation import DatasetCreator, ImageTagger
from src.modules.similarity_retrieval import (
    SimilarityRetriever, 
    EdgeSimilarityRetriever,
    CornerSimilarityRetriever
)
from src.modules.image_annotation import ImageAnnotator, AnnotationType


class LandmarkAnalysisApp:
    """
    Main application class for Visual Landmark and Scene Analysis.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the application.
        
        Args:
            data_dir: Directory for storing data (models, datasets, etc.)
        """
        # Set data directory
        if data_dir is None:
            # Use default data directory
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        else:
            self.data_dir = data_dir
        
        # Create data directories if they don't exist
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.models_dir = os.path.join(self.data_dir, 'models')
        self.datasets_dir = os.path.join(self.data_dir, 'datasets')
        self.annotations_dir = os.path.join(self.data_dir, 'annotations')
        self.results_dir = os.path.join(self.data_dir, 'results')
        
        for directory in [self.images_dir, self.models_dir, self.datasets_dir, 
                         self.annotations_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize modules
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.landmark_recognizer = LandmarkRecognizer()
        self.time_classifier = TimeOfDayClassifier()
        self.quality_assessor = ImageQualityAssessor()
        self.dataset_creator = DatasetCreator(self.datasets_dir)
        self.image_tagger = ImageTagger()
        self.similarity_retriever = SimilarityRetriever(feature_type='combined')
        self.edge_similarity_retriever = EdgeSimilarityRetriever()
        self.corner_similarity_retriever = CornerSimilarityRetriever()
        self.image_annotator = ImageAnnotator()
        
        # Load models if available
        self._load_models()
    
    def _load_models(self) -> None:
        """
        Load pre-trained models if available.
        """
        # Face recognition model
        face_model_path = os.path.join(self.models_dir, 'face_recognition_model.xml')
        if os.path.exists(face_model_path):
            try:
                self.face_recognizer.load_model(face_model_path)
                print(f"Loaded face recognition model from {face_model_path}")
            except Exception as e:
                print(f"Error loading face recognition model: {e}")
        
        # Landmark recognition database
        landmark_db_path = os.path.join(self.models_dir, 'landmark_database')
        if os.path.exists(landmark_db_path):    
            try:
                self.landmark_recognizer.load_database(landmark_db_path)
                print(f"Loaded landmark database from {landmark_db_path}")
            except Exception as e:
                print(f"Error loading landmark database: {e}")
        
        # Similarity retriever database
        similarity_db_path = os.path.join(self.models_dir, 'similarity_database.pkl')
        if os.path.exists(similarity_db_path):
            try:
                self.similarity_retriever.load_database(similarity_db_path)
                print(f"Loaded similarity database from {similarity_db_path}")
            except Exception as e:
                print(f"Error loading similarity database: {e}")
    
    def detect_faces(self, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image (optional)
            
        Returns:
            Tuple of (image with faces drawn, list of face bounding boxes)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        # Draw faces on the image
        result = self.face_detector.draw_faces(image, faces)
        
        # Save result if output path is provided
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result)
        
        return result, faces
    
    def recognize_landmark(self, image_path: str, output_path: Optional[str] = None) -> Tuple[Optional[str], int, Optional[np.ndarray]]:
        """
        Recognize a landmark in an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image (optional)
            
        Returns:
            Tuple of (landmark_name, match_count, result_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Recognize landmark
        landmark_name, match_count, result_image = self.landmark_recognizer.recognize_landmark(image)
        
        # Save result if output path is provided
        if output_path is not None and result_image is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result_image)
        
        return landmark_name, match_count, result_image
    
    def classify_time_of_day(self, image_path: str, output_path: Optional[str] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify the time of day in an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the visualization (optional)
            
        Returns:
            Tuple of (classification, confidence, metrics)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Classify time of day
        classification, confidence, metrics = self.time_classifier.classify(image)
        
        # Create visualization
        vis_image = self.time_classifier.visualize_classification(image, output_path)
        
        return classification, confidence, metrics
    
    def assess_image_quality(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess the quality of an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the visualization (optional)
            
        Returns:
            Dictionary containing assessment results and metrics
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Assess quality
        assessment = self.quality_assessor.assess_quality(image)
        
        # Create visualization
        vis_image = self.quality_assessor.visualize_assessment(image, assessment, output_path)
        
        return assessment
    
    def enhance_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Enhance an image based on quality assessment.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the enhanced image (optional)
            
        Returns:
            Enhanced image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Assess quality
        assessment = self.quality_assessor.assess_quality(image)
        
        # Enhance image
        enhanced = self.quality_assessor.enhance_image(image, assessment)
        
        # Save result if output path is provided
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, enhanced)
        
        return enhanced
    
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
        return self.dataset_creator.create_dataset(name, dataset_type, description)
    
    def add_images_to_dataset(self, dataset_id: str, directory: str, 
                            class_from_subdirs: bool = True) -> List[str]:
        """
        Add images from a directory to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            directory: Directory containing images
            class_from_subdirs: Whether to use subdirectory names as class names
            
        Returns:
            List of added image IDs
        """
        return self.dataset_creator.add_images_from_directory(
            dataset_id, directory, class_from_subdirs
        )
    
    def tag_dataset_images(self, dataset_id: str) -> None:
        """
        Automatically tag images in a dataset.
        
        Args:
            dataset_id: ID of the dataset
        """
        # Get dataset info
        dataset_info = self.dataset_creator.get_dataset_info(dataset_id)
        
        # Process each image
        for image_id in dataset_info['tags'].keys():
            try:
                # Get image path
                image_path = self.dataset_creator.get_image_path(dataset_id, image_id)
                
                # Load image
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Generate tags
                    tags = self.image_tagger.tag_all(image)
                    
                    # Add tags to image
                    self.dataset_creator.tag_image(dataset_id, image_id, {
                        'automatic_tags': tags
                    })
                    
                    print(f"Tagged image {image_id}")
            except Exception as e:
                print(f"Error tagging image {image_id}: {e}")
    
    def find_similar_images(self, image_path: str, similarity_type: str = 'combined', 
                          top_k: int = 3, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find images similar to the input image.
        
        Args:
            image_path: Path to the input image
            similarity_type: Type of similarity to use ('combined', 'edge', 'corner')
            top_k: Number of similar images to retrieve
            output_path: Path to save the visualization (optional)
            
        Returns:
            List of dictionaries with similar image information
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Select retriever based on similarity type
        if similarity_type == 'edge':
            retriever = self.edge_similarity_retriever
        elif similarity_type == 'corner':
            retriever = self.corner_similarity_retriever
        else:
            retriever = self.similarity_retriever
        
        # Find similar images
        similar_images = retriever.retrieve_similar_images(image, top_k)
        
        # Create visualization
        if similar_images and output_path is not None:
            vis_image = retriever.visualize_similar_images(image, similar_images, output_path)
        
        return similar_images
    
    def annotate_image(self, image_path: str, annotations_path: Optional[str] = None, 
                     output_path: Optional[str] = None) -> None:
        """
        Annotate an image.
        
        Args:
            image_path: Path to the input image
            annotations_path: Path to save the annotations (optional)
            output_path: Path to save the annotated image (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Start annotation
        annotations = self.image_annotator.annotate_image(image)
        
        # Save annotations if path is provided
        if annotations_path is not None:
            os.makedirs(os.path.dirname(annotations_path), exist_ok=True)
            self.image_annotator.save_annotations(annotations, annotations_path)
        
        # Save annotated image if path is provided
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.image_annotator.save_annotated_image(image, annotations, output_path)
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image with all available functionality.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary with processing results
        """
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Initialize results dictionary
        results = {
            'image_path': image_path,
            'output_dir': output_dir
        }
        
        # Detect faces
        try:
            face_output_path = os.path.join(output_dir, 'faces.jpg') if output_dir else None
            face_image, faces = self.detect_faces(image_path, face_output_path)
            results['faces'] = {
                'count': len(faces),
                'locations': faces,
                'output_path': face_output_path
            }
            print(f"Detected {len(faces)} faces")
        except Exception as e:
            print(f"Error detecting faces: {e}")
            results['faces'] = {'error': str(e)}
        
        # Recognize landmark
        try:
            landmark_output_path = os.path.join(output_dir, 'landmark.jpg') if output_dir else None
            landmark_name, match_count, landmark_image = self.landmark_recognizer.recognize_landmark(image)
            if landmark_image is not None and landmark_output_path:
                cv2.imwrite(landmark_output_path, landmark_image)
            results['landmark'] = {
                'name': landmark_name,
                'match_count': match_count,
                'output_path': landmark_output_path if landmark_image is not None else None
            }
            print(f"Landmark recognition: {landmark_name if landmark_name else 'Unknown'}")
        except Exception as e:
            print(f"Error recognizing landmark: {e}")
            results['landmark'] = {'error': str(e)}
        
        # Classify time of day
        try:
            tod_output_path = os.path.join(output_dir, 'time_of_day.jpg') if output_dir else None
            classification, confidence, metrics = self.time_classifier.classify(image)
            vis_image = self.time_classifier.visualize_classification(image, tod_output_path)
            results['time_of_day'] = {
                'classification': classification,
                'confidence': confidence,
                'output_path': tod_output_path
            }
            print(f"Time of day: {classification} (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error classifying time of day: {e}")
            results['time_of_day'] = {'error': str(e)}
        
        # Assess image quality
        try:
            quality_output_path = os.path.join(output_dir, 'quality.jpg') if output_dir else None
            assessment = self.quality_assessor.assess_quality(image)
            vis_image = self.quality_assessor.visualize_assessment(image, assessment, quality_output_path)
            
            # Highlight issues
            if assessment['quality_issues']:
                highlight_output_path = os.path.join(output_dir, 'quality_highlighted.jpg') if output_dir else None
                highlighted = self.quality_assessor.highlight_issues(image, assessment)
                if highlight_output_path:
                    cv2.imwrite(highlight_output_path, highlighted)
            else:
                highlight_output_path = None
                highlighted = None
            
            results['quality'] = {
                'overall_quality': assessment['overall_quality'],
                'issues': assessment['quality_issues'],
                'suggestions': assessment['improvement_suggestions'],
                'output_path': quality_output_path,
                'highlighted_path': highlight_output_path
            }
            print(f"Quality assessment: {assessment['overall_quality']}")
            if assessment['quality_issues']:
                print(f"  Issues: {', '.join(assessment['quality_issues'])}")
                print(f"  Suggestions: {', '.join(assessment['improvement_suggestions'])}")
        except Exception as e:
            print(f"Error assessing quality: {e}")
            results['quality'] = {'error': str(e)}
        
        # Find similar images
        try:
            similar_output_path = os.path.join(output_dir, 'similar.jpg') if output_dir else None
            similar_images = self.similarity_retriever.retrieve_similar_images(image, 3)
            if similar_images:
                vis_image = self.similarity_retriever.visualize_similar_images(image, similar_images, similar_output_path)
                results['similar'] = {
                    'count': len(similar_images),
                    'images': [{'path': img['path'], 'score': img['similarity_score']} for img in similar_images],
                    'output_path': similar_output_path
                }
                print(f"Found {len(similar_images)} similar images")
            else:
                results['similar'] = {
                    'count': 0,
                    'images': [],
                    'output_path': None
                }
                print("No similar images found")
        except Exception as e:
            print(f"Error finding similar images: {e}")
            results['similar'] = {'error': str(e)}
        
        return results


def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visual Landmark and Scene Analysis Application')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process image command
    process_parser = subparsers.add_parser('process', help='Process an image with all functionality')
    process_parser.add_argument('image', help='Path to the input image')
    process_parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    
    # Detect faces command
    faces_parser = subparsers.add_parser('faces', help='Detect faces in an image')
    faces_parser.add_argument('image', help='Path to the input image')
    faces_parser.add_argument('--output', '-o', help='Path to save the output image')
    
    # Recognize landmark command
    landmark_parser = subparsers.add_parser('landmark', help='Recognize a landmark in an image')
    landmark_parser.add_argument('image', help='Path to the input image')
    landmark_parser.add_argument('--output', '-o', help='Path to save the output image')
    
    # Classify time of day command
    time_parser = subparsers.add_parser('time', help='Classify the time of day in an image')
    time_parser.add_argument('image', help='Path to the input image')
    time_parser.add_argument('--output', '-o', help='Path to save the visualization')
    
    # Assess image quality command
    quality_parser = subparsers.add_parser('quality', help='Assess the quality of an image')
    quality_parser.add_argument('image', help='Path to the input image')
    quality_parser.add_argument('--output', '-o', help='Path to save the visualization')
    
    # Enhance image command
    enhance_parser = subparsers.add_parser('enhance', help='Enhance an image based on quality assessment')
    enhance_parser.add_argument('image', help='Path to the input image')
    enhance_parser.add_argument('--output', '-o', help='Path to save the enhanced image')
    
    # Create dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Create a new dataset')
    dataset_parser.add_argument('name', help='Name of the dataset')
    dataset_parser.add_argument('--type', '-t', default='general', help='Type of the dataset')
    dataset_parser.add_argument('--description', '-d', help='Description of the dataset')
    
    # Add images to dataset command
    add_images_parser = subparsers.add_parser('add-images', help='Add images to a dataset')
    add_images_parser.add_argument('dataset_id', help='ID of the dataset')
    add_images_parser.add_argument('directory', help='Directory containing images')
    add_images_parser.add_argument('--no-class-from-subdirs', action='store_false', dest='class_from_subdirs',
                                 help='Do not use subdirectory names as class names')
    
    # Tag dataset images command
    tag_parser = subparsers.add_parser('tag', help='Automatically tag images in a dataset')
    tag_parser.add_argument('dataset_id', help='ID of the dataset')
    
    # Find similar images command
    similar_parser = subparsers.add_parser('similar', help='Find images similar to the input image')
    similar_parser.add_argument('image', help='Path to the input image')
    similar_parser.add_argument('--type', '-t', default='combined', choices=['combined', 'edge', 'corner'],
                              help='Type of similarity to use')
    similar_parser.add_argument('--top-k', '-k', type=int, default=3, help='Number of similar images to retrieve')
    similar_parser.add_argument('--output', '-o', help='Path to save the visualization')
    
    # Annotate image command
    annotate_parser = subparsers.add_parser('annotate', help='Annotate an image')
    annotate_parser.add_argument('image', help='Path to the input image')
    annotate_parser.add_argument('--annotations', '-a', help='Path to save the annotations')
    annotate_parser.add_argument('--output', '-o', help='Path to save the annotated image')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create application instance
    app = LandmarkAnalysisApp()
    
    # Execute command
    if args.command == 'process':
        results = app.process_image(args.image, args.output_dir)
        print(f"Processing complete. Results saved to {args.output_dir}")
    
    elif args.command == 'faces':
        result, faces = app.detect_faces(args.image, args.output)
        print(f"Detected {len(faces)} faces")
        if args.output:
            print(f"Result saved to {args.output}")
    
    elif args.command == 'landmark':
        landmark_name, match_count, result_image = app.recognize_landmark(args.image, args.output)
        if landmark_name:
            print(f"Recognized landmark: {landmark_name} with {match_count} matches")
        else:
            print("No landmark recognized")
        if args.output and result_image is not None:
            print(f"Result saved to {args.output}")
    
    elif args.command == 'time':
        classification, confidence, metrics = app.classify_time_of_day(args.image, args.output)
        print(f"Time of day: {classification} (confidence: {confidence:.2f})")
        if args.output:
            print(f"Visualization saved to {args.output}")
    
    elif args.command == 'quality':
        assessment = app.assess_image_quality(args.image, args.output)
        print(f"Quality assessment: {assessment['overall_quality']}")
        if assessment['quality_issues']:
            print(f"Issues: {', '.join(assessment['quality_issues'])}")
            print(f"Suggestions: {', '.join(assessment['improvement_suggestions'])}")
        if args.output:
            print(f"Visualization saved to {args.output}")
    
    elif args.command == 'enhance':
        enhanced = app.enhance_image(args.image, args.output)
        if args.output:
            print(f"Enhanced image saved to {args.output}")
    
    elif args.command == 'dataset':
        dataset_id = app.create_dataset(args.name, args.type, args.description)
        print(f"Created dataset with ID: {dataset_id}")
    
    elif args.command == 'add-images':
        image_ids = app.add_images_to_dataset(args.dataset_id, args.directory, args.class_from_subdirs)
        print(f"Added {len(image_ids)} images to dataset {args.dataset_id}")
    
    elif args.command == 'tag':
        app.tag_dataset_images(args.dataset_id)
        print(f"Tagged images in dataset {args.dataset_id}")
    
    elif args.command == 'similar':
        similar_images = app.find_similar_images(args.image, args.type, args.top_k, args.output)
        if similar_images:
            print(f"Found {len(similar_images)} similar images:")
            for i, similar in enumerate(similar_images):
                print(f"  {i+1}. {similar['path']} (Similarity: {similar['similarity_score']:.2f})")
        else:
            print("No similar images found")
        if args.output:
            print(f"Visualization saved to {args.output}")
    
    elif args.command == 'annotate':
        app.annotate_image(args.image, args.annotations, args.output)
        if args.annotations:
            print(f"Annotations saved to {args.annotations}")
        if args.output:
            print(f"Annotated image saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
