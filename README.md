# Visual Landmark and Scene Analysis Application

This application provides a comprehensive toolkit for visual landmark and scene analysis, including face recognition, landmark recognition, time-of-day classification, image quality assessment, dataset creation, similarity retrieval, and image annotation.

## Features

- **Landmark and Face Recognition**: Detect and identify landmarks and human faces in images
- **Time-of-Day Classification**: Determine whether images were captured during daytime or nighttime
- **Image Quality Assessment**: Analyze image quality and suggest potential enhancements
- **Dataset Creation**: Create and manage datasets of images with proper tagging
- **Similarity Retrieval**: Find images similar to a query image based on various similarity metrics
- **Image Annotation**: Draw annotations directly on still images

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.5.0 or higher (with contrib modules for additional features)
- NumPy, Matplotlib, and other dependencies listed in `requirements.txt`

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/landmark-analysis-app.git
   cd landmark-analysis-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - For face recognition: Place face images in `data/images/faces` with subdirectories for each person
   - For landmark recognition: Place landmark images in `data/images/landmarks` with subdirectories for each landmark

## Usage

The application provides both a command-line interface and a Python API for integration into your projects.

### Command-Line Interface

Process an image with all functionality:
```
python src/main.py process path/to/image.jpg --output-dir results/
```

Detect faces in an image:
```
python src/main.py faces path/to/image.jpg --output results/faces.jpg
```

Recognize landmarks:
```
python src/main.py landmark path/to/image.jpg --output results/landmark.jpg
```

Classify time of day:
```
python src/main.py time path/to/image.jpg --output results/time.jpg
```

Assess image quality:
```
python src/main.py quality path/to/image.jpg --output results/quality.jpg
```

Enhance an image:
```
python src/main.py enhance path/to/image.jpg --output results/enhanced.jpg
```

Find similar images:
```
python src/main.py similar path/to/image.jpg --type combined --top-k 3 --output results/similar.jpg
```

Annotate an image:
```
python src/main.py annotate path/to/image.jpg --annotations results/annotations.json --output results/annotated.jpg
```

Create and manage datasets:
```
python src/main.py dataset "Dubai Landmarks" --type landmarks --description "Dataset of landmarks in Dubai"
python src/main.py add-images dataset_id path/to/images/
python src/main.py tag dataset_id
```

### Python API

```python
from src.main import LandmarkAnalysisApp

# Initialize the application
app = LandmarkAnalysisApp()

# Process an image
results = app.process_image('path/to/image.jpg', 'results/')

# Detect faces
face_image, faces = app.detect_faces('path/to/image.jpg', 'results/faces.jpg')

# Recognize landmarks
landmark_name, match_count, result_image = app.recognize_landmark('path/to/image.jpg', 'results/landmark.jpg')

# Classify time of day
classification, confidence, metrics = app.classify_time_of_day('path/to/image.jpg', 'results/time.jpg')

# Assess image quality
assessment = app.assess_image_quality('path/to/image.jpg', 'results/quality.jpg')

# Enhance an image
enhanced = app.enhance_image('path/to/image.jpg', 'results/enhanced.jpg')

# Find similar images
similar_images = app.find_similar_images('path/to/image.jpg', 'combined', 3, 'results/similar.jpg')

# Annotate an image
app.annotate_image('path/to/image.jpg', 'results/annotations.json', 'results/annotated.jpg')
```

## Module Documentation

### Face Recognition

The face recognition module uses Haar cascade classifiers for face detection and Local Binary Pattern Histograms (LBPH) for face recognition.

```python
from src.modules.face_recognition import FaceDetector, FaceRecognizer

# Detect faces
detector = FaceDetector()
faces = detector.detect_faces(image)

# Recognize faces
recognizer = FaceRecognizer()
recognizer.train(faces, labels, label_names)
label, confidence, name = recognizer.predict(face)
```

### Landmark Recognition

The landmark recognition module uses ORB features and feature matching for landmark recognition.

```python
from src.modules.landmark_recognition import LandmarkRecognizer

# Initialize recognizer
recognizer = LandmarkRecognizer()

# Add landmarks to database
recognizer.add_landmark('Burj Khalifa', image)

# Recognize landmark
landmark_name, match_count, result_image = recognizer.recognize_landmark(image)
```

### Time-of-Day Classification

The time-of-day classification module uses multiple color space approaches (HSV, LAB, grayscale) to determine whether an image was captured during daytime or nighttime.

```python
from src.modules.time_of_day_classification import TimeOfDayClassifier

# Initialize classifier
classifier = TimeOfDayClassifier()

# Classify time of day
classification, confidence, metrics = classifier.classify(image)

# Visualize classification
vis_image = classifier.visualize_classification(image, 'results/time.jpg')
```

### Image Quality Assessment

The image quality assessment module analyzes image quality and suggests potential enhancements.

```python
from src.modules.image_quality_assessment import ImageQualityAssessor

# Initialize assessor
assessor = ImageQualityAssessor()

# Assess quality
assessment = assessor.assess_quality(image)

# Visualize assessment
vis_image = assessor.visualize_assessment(image, assessment, 'results/quality.jpg')

# Enhance image
enhanced = assessor.enhance_image(image, assessment)
```

### Dataset Creation

The dataset creation module provides functionality for creating and managing datasets of images with proper tagging.

```python
from src.modules.dataset_creation import DatasetCreator, ImageTagger

# Initialize creator and tagger
creator = DatasetCreator('data/datasets')
tagger = ImageTagger()

# Create dataset
dataset_id = creator.create_dataset('Dubai Landmarks', 'landmarks', 'Dataset of landmarks in Dubai')

# Add images
creator.add_class(dataset_id, 'burj_khalifa', 'Burj Khalifa')
image_ids = creator.add_images_from_directory(dataset_id, 'data/images/landmarks')

# Tag images
tags = tagger.tag_all(image)
creator.tag_image(dataset_id, image_id, {'automatic_tags': tags})
```

### Similarity Retrieval

The similarity retrieval module finds images similar to a query image based on various similarity metrics.

```python
from src.modules.similarity_retrieval import SimilarityRetriever, EdgeSimilarityRetriever, CornerSimilarityRetriever

# Initialize retrievers
retriever = SimilarityRetriever(feature_type='combined')
edge_retriever = EdgeSimilarityRetriever()
corner_retriever = CornerSimilarityRetriever()

# Add images to database
retriever.add_images_from_directory('data/images/landmarks')

# Find similar images
similar_images = retriever.retrieve_similar_images(image, 3)

# Visualize similar images
vis_image = retriever.visualize_similar_images(image, similar_images, 'results/similar.jpg')
```

### Image Annotation

The image annotation module allows drawing annotations directly on still images.

```python
from src.modules.image_annotation import ImageAnnotator, AnnotationType

# Initialize annotator
annotator = ImageAnnotator()

# Annotate image
annotations = annotator.annotate_image(image)

# Save annotations
annotator.save_annotations(annotations, 'results/annotations.json')

# Save annotated image
annotator.save_annotated_image(image, annotations, 'results/annotated.jpg')
```

## Example Workflows

### Complete Image Analysis

```python
from src.main import LandmarkAnalysisApp

# Initialize the application
app = LandmarkAnalysisApp()

# Process an image with all functionality
results = app.process_image('path/to/image.jpg', 'results/')

# Print results
print(f"Detected {results['faces']['count']} faces")
print(f"Landmark: {results['landmark']['name']}")
print(f"Time of day: {results['time_of_day']['classification']}")
print(f"Quality: {results['quality']['overall_quality']}")
if results['quality']['issues']:
    print(f"Issues: {', '.join(results['quality']['issues'])}")
    print(f"Suggestions: {', '.join(results['quality']['suggestions'])}")
print(f"Found {results['similar']['count']} similar images")
```

### Dataset Creation and Tagging

```python
from src.main import LandmarkAnalysisApp

# Initialize the application
app = LandmarkAnalysisApp()

# Create dataset
dataset_id = app.create_dataset('Dubai Landmarks', 'landmarks', 'Dataset of landmarks in Dubai')

# Add images
image_ids = app.add_images_to_dataset(dataset_id, 'data/images/landmarks')

# Tag images
app.tag_dataset_images(dataset_id)
```

## Adding Your Own Dubai Landmark Dataset

To use this application with your own Dubai landmark dataset:

1. Organize your images in a directory structure:
   ```
   data/images/landmarks/
   ├── burj_khalifa/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── dubai_marina/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```

2. Create a dataset:
   ```
   python src/main.py dataset "Dubai Landmarks" --type landmarks --description "Dataset of landmarks in Dubai"
   ```

3. Add your images:
   ```
   python src/main.py add-images dataset_id data/images/landmarks/
   ```

4. Tag the images:
   ```
   python src/main.py tag dataset_id
   ```

5. Test with a query image:
   ```
   python src/main.py process path/to/query.jpg --output-dir results/
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for providing the computer vision algorithms
- The creators of the various datasets used for testing
#   l a n d m a r k _ a n a l y s i s _ a p p  
 