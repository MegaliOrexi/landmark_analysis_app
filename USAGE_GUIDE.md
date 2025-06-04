# Visual Landmark and Scene Analysis Application - Usage Guide

This guide provides detailed instructions on how to use the Visual Landmark and Scene Analysis Application for analyzing images of landmarks, particularly for Dubai landmarks.

## Getting Started

### Installation

1. Ensure you have Python 3.8 or higher installed on your system.

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset structure (see "Adding Your Own Dataset" section below).

### Basic Usage

The application can be used either through the command-line interface or by importing the modules in your Python code.

## Command-Line Interface

### Process an Image with All Functionality

This command runs all analysis modules on a single image:

```bash
python src/main.py process path/to/image.jpg --output-dir results/
```

This will:
- Detect and recognize faces
- Identify landmarks
- Classify time of day (day/night)
- Assess image quality
- Find similar images
- Save all results to the specified output directory

### Individual Module Commands

#### Face Detection

```bash
python src/main.py faces path/to/image.jpg --output results/faces.jpg
```

#### Landmark Recognition

```bash
python src/main.py landmark path/to/image.jpg --output results/landmark.jpg
```

#### Time-of-Day Classification

```bash
python src/main.py time path/to/image.jpg --output results/time.jpg
```

#### Image Quality Assessment

```bash
python src/main.py quality path/to/image.jpg --output results/quality.jpg
```

#### Image Enhancement

```bash
python src/main.py enhance path/to/image.jpg --output results/enhanced.jpg
```

#### Find Similar Images

```bash
python src/main.py similar path/to/image.jpg --type combined --top-k 3 --output results/similar.jpg
```

Options for `--type`:
- `combined`: Uses a combination of feature matching, color histograms, and HOG features
- `edge`: Focuses on edge features for similarity
- `corner`: Focuses on corner features for similarity

#### Image Annotation

```bash
python src/main.py annotate path/to/image.jpg --annotations results/annotations.json --output results/annotated.jpg
```

When using the annotation tool:
- Press 'r' for rectangle, 'c' for circle, 'l' for line, 'a' for arrow
- Press 't' for text, 'p' for polygon, 'f' for freehand drawing
- Press '1'-'9' to set a label, '0' to clear the label
- Press '+'/'-' to increase/decrease line thickness
- Press SPACE to toggle filled/outline mode
- Press DELETE to remove the last annotation
- Press ESC to finish annotation

### Dataset Management

#### Create a New Dataset

```bash
python src/main.py dataset "Dubai Landmarks" --type landmarks --description "Dataset of landmarks in Dubai"
```

This will return a dataset ID that you'll use in subsequent commands.

#### Add Images to a Dataset

```bash
python src/main.py add-images dataset_id path/to/images/
```

By default, the application will use subdirectory names as class names. Use `--no-class-from-subdirs` to disable this behavior.

#### Tag Images in a Dataset

```bash
python src/main.py tag dataset_id
```

This will automatically analyze and tag all images in the dataset with information about faces, colors, features, time of day, and quality.

## Python API

You can also use the application programmatically in your Python code:

```python
from src.main import LandmarkAnalysisApp

# Initialize the application
app = LandmarkAnalysisApp()

# Process an image with all functionality
results = app.process_image('path/to/image.jpg', 'results/')

# Access specific results
print(f"Detected {results['faces']['count']} faces")
print(f"Landmark: {results['landmark']['name']}")
print(f"Time of day: {results['time_of_day']['classification']}")
print(f"Quality: {results['quality']['overall_quality']}")
```

## Adding Your Own Dataset

### For Landmark Recognition

1. Create a directory structure where each landmark has its own subdirectory:
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
   ```bash
   python src/main.py dataset "Dubai Landmarks" --type landmarks --description "Dataset of landmarks in Dubai"
   ```
   Note the dataset ID returned by this command.

3. Add your images:
   ```bash
   python src/main.py add-images dataset_id data/images/landmarks/
   ```

4. Tag the images:
   ```bash
   python src/main.py tag dataset_id
   ```

### For Face Recognition

1. Create a directory structure where each person has their own subdirectory:
   ```
   data/images/faces/
   ├── person1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── person2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```

2. Create a dataset:
   ```bash
   python src/main.py dataset "Face Dataset" --type faces --description "Dataset of faces"
   ```

3. Add your images:
   ```bash
   python src/main.py add-images dataset_id data/images/faces/
   ```

4. Tag the images:
   ```bash
   python src/main.py tag dataset_id
   ```

## Advanced Usage

### Calibrating Time-of-Day Classification

If you want to calibrate the time-of-day classification for your specific dataset:

```python
from src.modules.time_of_day_classification import TimeOfDayClassifier
import cv2
import os

# Initialize classifier
classifier = TimeOfDayClassifier()

# Load day and night images
day_images = []
night_images = []

day_dir = "data/images/time_of_day/day"
night_dir = "data/images/time_of_day/night"

for filename in os.listdir(day_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(day_dir, filename))
        if img is not None:
            day_images.append(img)

for filename in os.listdir(night_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(night_dir, filename))
        if img is not None:
            night_images.append(img)

# Calibrate thresholds
calibrated_thresholds = classifier.calibrate_thresholds(day_images, night_images)
print(f"Calibrated thresholds: {calibrated_thresholds}")
```

### Creating a Custom Similarity Database

To create a custom database for similarity retrieval:

```python
from src.modules.similarity_retrieval import SimilarityRetriever
import os

# Initialize retriever
retriever = SimilarityRetriever(feature_type='combined')

# Add images from directory
retriever.add_images_from_directory('data/images/landmarks', recursive=True)

# Save database
retriever.save_database('data/models/similarity_database.pkl')
```

## Troubleshooting

### Common Issues

1. **OpenCV Installation Issues**: If you encounter problems with OpenCV, try installing the base and contrib modules separately:
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python
   pip install opencv-contrib-python
   ```

2. **SIFT Not Available**: If you see warnings about SIFT not being available, ensure you have installed opencv-contrib-python.

3. **Memory Issues with Large Datasets**: When processing large datasets, you might encounter memory issues. Process images in smaller batches:
   ```python
   # Instead of:
   retriever.add_images_from_directory('data/images/landmarks')
   
   # Use:
   for subdir in os.listdir('data/images/landmarks'):
       subdir_path = os.path.join('data/images/landmarks', subdir)
       if os.path.isdir(subdir_path):
           print(f"Processing {subdir}...")
           retriever.add_images_from_directory(subdir_path)
   ```

4. **Slow Performance**: For better performance with large datasets, consider:
   - Reducing image resolution before processing
   - Using a smaller number of features (e.g., `orb = cv2.ORB_create(nfeatures=500)` instead of 1000)
   - Using simpler similarity metrics (e.g., 'color_hist' instead of 'combined')

## Performance Optimization

For optimal performance, especially with Dubai landmark images:

1. **Image Resolution**: Resize large images to a maximum width/height of 1024px before processing.

2. **Feature Extraction**: For landmark recognition, adjust the number of features based on the complexity of your landmarks:
   ```python
   # For detailed landmarks like Burj Khalifa
   recognizer = LandmarkRecognizer(max_features=1500)
   
   # For simpler landmarks
   recognizer = LandmarkRecognizer(max_features=800)
   ```

3. **Time-of-Day Classification**: Dubai has unique lighting conditions. You might want to adjust the thresholds:
   ```python
   classifier = TimeOfDayClassifier()
   classifier.set_thresholds({
       'hsv_value': 90.0,  # Higher threshold for bright Dubai daylight
       'lab_lightness': 60.0
   })
   ```

4. **Quality Assessment**: For desert environments with high brightness:
   ```python
   assessor = ImageQualityAssessor()
   assessor.set_thresholds({
       'brightness_high': 220.0,  # Higher threshold for bright conditions
       'saturation': 0.07  # Higher threshold for colorful scenes
   })
   ```

## Example Workflows

### Complete Analysis Pipeline

```python
from src.main import LandmarkAnalysisApp
import os

# Initialize application
app = LandmarkAnalysisApp()

# Process a directory of images
input_dir = "data/input_images"
output_base = "data/results"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing {filename}...")
        image_path = os.path.join(input_dir, filename)
        output_dir = os.path.join(output_base, os.path.splitext(filename)[0])
        
        # Process image
        results = app.process_image(image_path, output_dir)
        
        # Print summary
        print(f"  Faces: {results['faces']['count']}")
        print(f"  Landmark: {results['landmark']['name'] if results['landmark']['name'] else 'Unknown'}")
        print(f"  Time: {results['time_of_day']['classification']}")
        print(f"  Quality: {results['quality']['overall_quality']}")
```

### Creating a Custom Dataset and Testing Recognition

```python
from src.main import LandmarkAnalysisApp
import os

# Initialize application
app = LandmarkAnalysisApp()

# Create dataset
dataset_id = app.create_dataset("Dubai Landmarks", "landmarks", "Dataset of landmarks in Dubai")
print(f"Created dataset with ID: {dataset_id}")

# Add images
image_ids = app.add_images_to_dataset(dataset_id, "data/images/landmarks")
print(f"Added {len(image_ids)} images to dataset")

# Tag images
app.tag_dataset_images(dataset_id)
print("Tagged all images")

# Test with a query image
test_image = "data/test_images/unknown_landmark.jpg"
results = app.process_image(test_image, "data/results/test")
print(f"Recognized as: {results['landmark']['name'] if results['landmark']['name'] else 'Unknown'}")
```

## Further Assistance

If you encounter any issues or have questions about using the application with your Dubai landmark dataset, please refer to the module documentation or contact the developer for assistance.
