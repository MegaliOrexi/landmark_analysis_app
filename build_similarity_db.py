#!/usr/bin/env python3
"""
Build similarity database for image retrieval.

This script creates a database of image features for similarity retrieval
using various feature extraction methods (ORB, color histogram, HOG, etc.).
"""

import os
import sys
import argparse
from src.modules.similarity_retrieval import SimilarityRetriever

def build_similarity_database(image_directories, output_path="data/models/similarity_database.pkl", 
                            feature_type="combined", recursive=True):
    """
    Build a similarity database from multiple image directories.
    
    Args:
        image_directories: List of directories containing images
        output_path: Path to save the similarity database
        feature_type: Type of features to use ('orb', 'combined', 'color_hist', 'hog')
        recursive: Whether to search subdirectories recursively
    """
    # Initialize the similarity retriever
    retriever = SimilarityRetriever(feature_type=feature_type)
    
    total_images = 0
    
    # Process each directory
    for directory in image_directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
            
        print(f"Processing directory: {directory}")
        
        # Add images from directory
        try:
            count = retriever.add_images_from_directory(directory, recursive=recursive)
            total_images += count
            print(f"Added {count} images from {directory}")
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")
    
    if total_images == 0:
        print("No images were added to the database")
        return False
    
    # Save the similarity database
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    retriever.save_database(output_path)
    print(f"Similarity database saved to: {output_path}")
    print(f"Total images in database: {total_images}")
    
    return True

def main():
    """
    Main entry point for building similarity database.
    """
    parser = argparse.ArgumentParser(description='Build similarity database for image retrieval')
    parser.add_argument('--directories', '-d', nargs='+', 
                       default=['data/images/landmarks', 'data/extraImages'],
                       help='Directories containing images to add to database')
    parser.add_argument('--output', '-o', default='data/models/similarity_database.pkl',
                       help='Path to save the similarity database')
    parser.add_argument('--feature-type', '-f', default='combined',
                       choices=['orb', 'combined', 'color_hist', 'hog'],
                       help='Type of features to extract')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                       help='Do not search subdirectories recursively')
    
    args = parser.parse_args()
    
    # Build the database
    success = build_similarity_database(
        args.directories, 
        args.output, 
        args.feature_type, 
        args.recursive
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 