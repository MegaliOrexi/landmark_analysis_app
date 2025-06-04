import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import uuid
from datetime import datetime


class AnnotationType(Enum):
    """
    Enumeration of annotation types.
    """
    RECTANGLE = 'rectangle'
    CIRCLE = 'circle'
    LINE = 'line'
    ARROW = 'arrow'
    TEXT = 'text'
    POLYGON = 'polygon'
    FREEHAND = 'freehand'


class Annotation:
    """
    Class representing an annotation.
    """
    
    def __init__(self, annotation_type: AnnotationType, points: List[Tuple[int, int]], 
                label: Optional[str] = None, color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2, filled: bool = False, font_scale: float = 1.0):
        """
        Initialize an annotation.
        
        Args:
            annotation_type: Type of annotation
            points: List of points defining the annotation
            label: Optional label for the annotation
            color: Color of the annotation (BGR format)
            thickness: Thickness of the annotation lines
            filled: Whether the annotation should be filled
            font_scale: Scale of the font for text annotations
        """
        self.id = str(uuid.uuid4())
        self.type = annotation_type
        self.points = points
        self.label = label
        self.color = color
        self.thickness = thickness
        self.filled = filled
        self.font_scale = font_scale
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the annotation to a dictionary.
        
        Returns:
            Dictionary representation of the annotation
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'points': self.points,
            'label': self.label,
            'color': self.color,
            'thickness': self.thickness,
            'filled': self.filled,
            'font_scale': self.font_scale,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Annotation':
        """
        Create an annotation from a dictionary.
        
        Args:
            data: Dictionary representation of the annotation
            
        Returns:
            Annotation object
        """
        # Create annotation
        annotation = cls(
            annotation_type=AnnotationType(data['type']),
            points=data['points'],
            label=data['label'],
            color=tuple(data['color']),
            thickness=data['thickness'],
            filled=data['filled'],
            font_scale=data['font_scale']
        )
        
        # Set ID and creation time
        annotation.id = data['id']
        annotation.created_at = data['created_at']
        
        return annotation
    
    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw the annotation on an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Image with annotation drawn
        """
        # Make a copy of the image
        img_copy = image.copy()
        
        # Draw based on annotation type
        if self.type == AnnotationType.RECTANGLE:
            # Check if we have at least 2 points
            if len(self.points) >= 2:
                pt1 = self.points[0]
                pt2 = self.points[1]
                
                # Draw rectangle
                if self.filled:
                    cv2.rectangle(img_copy, pt1, pt2, self.color, -1)
                else:
                    cv2.rectangle(img_copy, pt1, pt2, self.color, self.thickness)
        
        elif self.type == AnnotationType.CIRCLE:
            # Check if we have at least 2 points
            if len(self.points) >= 2:
                center = self.points[0]
                
                # Calculate radius as distance between points
                radius = int(np.sqrt((self.points[1][0] - center[0])**2 + (self.points[1][1] - center[1])**2))
                
                # Draw circle
                if self.filled:
                    cv2.circle(img_copy, center, radius, self.color, -1)
                else:
                    cv2.circle(img_copy, center, radius, self.color, self.thickness)
        
        elif self.type == AnnotationType.LINE:
            # Check if we have at least 2 points
            if len(self.points) >= 2:
                # Draw line
                cv2.line(img_copy, self.points[0], self.points[1], self.color, self.thickness)
        
        elif self.type == AnnotationType.ARROW:
            # Check if we have at least 2 points
            if len(self.points) >= 2:
                # Draw arrow
                cv2.arrowedLine(img_copy, self.points[0], self.points[1], self.color, self.thickness)
        
        elif self.type == AnnotationType.TEXT:
            # Check if we have at least 1 point and a label
            if len(self.points) >= 1 and self.label is not None:
                # Draw text
                cv2.putText(img_copy, self.label, self.points[0], cv2.FONT_HERSHEY_SIMPLEX, 
                          self.font_scale, self.color, self.thickness)
        
        elif self.type == AnnotationType.POLYGON:
            # Check if we have at least 3 points
            if len(self.points) >= 3:
                # Convert points to numpy array
                points = np.array(self.points, dtype=np.int32)
                
                # Draw polygon
                if self.filled:
                    cv2.fillPoly(img_copy, [points], self.color)
                else:
                    cv2.polylines(img_copy, [points], True, self.color, self.thickness)
        
        elif self.type == AnnotationType.FREEHAND:
            # Check if we have at least 2 points
            if len(self.points) >= 2:
                # Draw lines between consecutive points
                for i in range(len(self.points) - 1):
                    cv2.line(img_copy, self.points[i], self.points[i+1], self.color, self.thickness)
        
        # Draw label if present and not already drawn (for text annotations)
        if self.label is not None and self.type != AnnotationType.TEXT:
            # Determine label position
            if self.type == AnnotationType.RECTANGLE and len(self.points) >= 2:
                label_pos = (self.points[0][0], self.points[0][1] - 10)
            elif self.type == AnnotationType.CIRCLE and len(self.points) >= 2:
                label_pos = (self.points[0][0], self.points[0][1] - 10)
            else:
                # Use first point
                label_pos = self.points[0]
            
            # Draw label
            cv2.putText(img_copy, self.label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, self.color, 2)
        
        return img_copy


class ImageAnnotator:
    """
    Class for annotating images.
    """
    
    def __init__(self):
        """
        Initialize the image annotator.
        """
        # List of annotations
        self.annotations = []
        
        # Current annotation being created
        self.current_annotation = None
        
        # Current annotation type
        self.current_type = AnnotationType.RECTANGLE
        
        # Current annotation properties
        self.current_color = (0, 255, 0)  # Green
        self.current_thickness = 2
        self.current_filled = False
        self.current_font_scale = 1.0
        
        # Current label
        self.current_label = None
        
        # Drawing state
        self.drawing = False
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        
        # Image being annotated
        self.image = None
        self.display_image = None
    
    def set_annotation_type(self, annotation_type: AnnotationType) -> None:
        """
        Set the current annotation type.
        
        Args:
            annotation_type: Type of annotation to create
        """
        self.current_type = annotation_type
    
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """
        Set the current annotation color.
        
        Args:
            color: Color in BGR format
        """
        self.current_color = color
    
    def set_thickness(self, thickness: int) -> None:
        """
        Set the current annotation thickness.
        
        Args:
            thickness: Thickness of annotation lines
        """
        self.current_thickness = thickness
    
    def set_filled(self, filled: bool) -> None:
        """
        Set whether annotations should be filled.
        
        Args:
            filled: Whether to fill annotations
        """
        self.current_filled = filled
    
    def set_font_scale(self, font_scale: float) -> None:
        """
        Set the font scale for text annotations.
        
        Args:
            font_scale: Scale of the font
        """
        self.current_font_scale = font_scale
    
    def set_label(self, label: Optional[str]) -> None:
        """
        Set the current annotation label.
        
        Args:
            label: Label for annotations
        """
        self.current_label = label
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Callback function for mouse events.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
            param: Additional parameters
        """
        # Handle mouse events based on annotation type
        if self.current_type == AnnotationType.RECTANGLE:
            self._handle_rectangle(event, x, y, flags)
        elif self.current_type == AnnotationType.CIRCLE:
            self._handle_circle(event, x, y, flags)
        elif self.current_type == AnnotationType.LINE or self.current_type == AnnotationType.ARROW:
            self._handle_line(event, x, y, flags)
        elif self.current_type == AnnotationType.TEXT:
            self._handle_text(event, x, y, flags)
        elif self.current_type == AnnotationType.POLYGON:
            self._handle_polygon(event, x, y, flags)
        elif self.current_type == AnnotationType.FREEHAND:
            self._handle_freehand(event, x, y, flags)
    
    def _handle_rectangle(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for rectangle annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.drag_start = (x, y)
            
            # Create new annotation
            self.current_annotation = Annotation(
                annotation_type=AnnotationType.RECTANGLE,
                points=[(x, y), (x, y)],
                label=self.current_label,
                color=self.current_color,
                thickness=self.current_thickness,
                filled=self.current_filled
            )
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update end point
                self.drag_end = (x, y)
                
                # Update annotation
                if self.current_annotation is not None:
                    self.current_annotation.points[1] = (x, y)
                
                # Update display
                self.display_image = self.draw_annotations()
                cv2.imshow('Annotation', self.display_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            # Add annotation to list
            if self.current_annotation is not None:
                self.annotations.append(self.current_annotation)
                self.current_annotation = None
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
    
    def _handle_circle(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for circle annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.drag_start = (x, y)
            
            # Create new annotation
            self.current_annotation = Annotation(
                annotation_type=AnnotationType.CIRCLE,
                points=[(x, y), (x, y)],
                label=self.current_label,
                color=self.current_color,
                thickness=self.current_thickness,
                filled=self.current_filled
            )
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update end point
                self.drag_end = (x, y)
                
                # Update annotation
                if self.current_annotation is not None:
                    self.current_annotation.points[1] = (x, y)
                
                # Update display
                self.display_image = self.draw_annotations()
                cv2.imshow('Annotation', self.display_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            # Add annotation to list
            if self.current_annotation is not None:
                self.annotations.append(self.current_annotation)
                self.current_annotation = None
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
    
    def _handle_line(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for line and arrow annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.drag_start = (x, y)
            
            # Create new annotation
            self.current_annotation = Annotation(
                annotation_type=self.current_type,
                points=[(x, y), (x, y)],
                label=self.current_label,
                color=self.current_color,
                thickness=self.current_thickness
            )
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update end point
                self.drag_end = (x, y)
                
                # Update annotation
                if self.current_annotation is not None:
                    self.current_annotation.points[1] = (x, y)
                
                # Update display
                self.display_image = self.draw_annotations()
                cv2.imshow('Annotation', self.display_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            # Add annotation to list
            if self.current_annotation is not None:
                self.annotations.append(self.current_annotation)
                self.current_annotation = None
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
    
    def _handle_text(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for text annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we have a label
            if self.current_label is None or self.current_label == "":
                print("Please set a label for text annotation")
                return
            
            # Create new annotation
            annotation = Annotation(
                annotation_type=AnnotationType.TEXT,
                points=[(x, y)],
                label=self.current_label,
                color=self.current_color,
                thickness=self.current_thickness,
                font_scale=self.current_font_scale
            )
            
            # Add annotation to list
            self.annotations.append(annotation)
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
    
    def _handle_polygon(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for polygon annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # If not drawing, start a new polygon
            if not self.drawing:
                self.drawing = True
                
                # Create new annotation
                self.current_annotation = Annotation(
                    annotation_type=AnnotationType.POLYGON,
                    points=[(x, y)],
                    label=self.current_label,
                    color=self.current_color,
                    thickness=self.current_thickness,
                    filled=self.current_filled
                )
            else:
                # Add point to current polygon
                if self.current_annotation is not None:
                    self.current_annotation.points.append((x, y))
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_annotation is not None:
                # Create a temporary copy for preview
                temp_image = self.draw_annotations()
                
                # Draw line from last point to current mouse position
                if len(self.current_annotation.points) > 0:
                    last_point = self.current_annotation.points[-1]
                    cv2.line(temp_image, last_point, (x, y), self.current_color, self.current_thickness)
                
                # Display
                cv2.imshow('Annotation', temp_image)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish polygon
            if self.drawing and self.current_annotation is not None:
                # Check if we have at least 3 points
                if len(self.current_annotation.points) >= 3:
                    # Add annotation to list
                    self.annotations.append(self.current_annotation)
                
                # Reset
                self.current_annotation = None
                self.drawing = False
                
                # Update display
                self.display_image = self.draw_annotations()
                cv2.imshow('Annotation', self.display_image)
    
    def _handle_freehand(self, event: int, x: int, y: int, flags: int) -> None:
        """
        Handle mouse events for freehand annotations.
        
        Args:
            event: Mouse event
            x: x-coordinate
            y: y-coordinate
            flags: Event flags
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            
            # Create new annotation
            self.current_annotation = Annotation(
                annotation_type=AnnotationType.FREEHAND,
                points=[(x, y)],
                label=self.current_label,
                color=self.current_color,
                thickness=self.current_thickness
            )
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_annotation is not None:
                # Add point to current freehand drawing
                self.current_annotation.points.append((x, y))
                
                # Update display
                self.display_image = self.draw_annotations()
                cv2.imshow('Annotation', self.display_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            # Add annotation to list
            if self.current_annotation is not None:
                self.annotations.append(self.current_annotation)
                self.current_annotation = None
            
            # Update display
            self.display_image = self.draw_annotations()
            cv2.imshow('Annotation', self.display_image)
    
    def draw_annotations(self) -> np.ndarray:
        """
        Draw all annotations on the image.
        
        Returns:
            Image with annotations drawn
        """
        # Start with a copy of the original image
        result = self.image.copy()
        
        # Draw all annotations
        for annotation in self.annotations:
            result = annotation.draw(result)
        
        # Draw current annotation if any
        if self.current_annotation is not None:
            result = self.current_annotation.draw(result)
        
        return result
    
    def annotate_image(self, image: np.ndarray) -> List[Annotation]:
        """
        Start the annotation process for an image.
        
        Args:
            image: Image to annotate (BGR format)
            
        Returns:
            List of annotations
        """
        # Store the image
        self.image = image.copy()
        self.display_image = image.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Annotation')
        cv2.setMouseCallback('Annotation', self._mouse_callback)
        
        # Display instructions
        print("Annotation Controls:")
        print("  - ESC: Finish annotation")
        print("  - r: Set annotation type to Rectangle")
        print("  - c: Set annotation type to Circle")
        print("  - l: Set annotation type to Line")
        print("  - a: Set annotation type to Arrow")
        print("  - t: Set annotation type to Text")
        print("  - p: Set annotation type to Polygon (right-click to finish)")
        print("  - f: Set annotation type to Freehand")
        print("  - 1-9: Set label to corresponding number")
        print("  - 0: Clear label")
        print("  - +/-: Increase/decrease thickness")
        print("  - SPACE: Toggle filled/outline")
        print("  - DELETE: Remove last annotation")
        
        # Main loop
        while True:
            # Display the image with annotations
            cv2.imshow('Annotation', self.display_image)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Check for exit
            if key == 27:  # ESC
                break
            
            # Check for annotation type change
            elif key == ord('r'):
                self.set_annotation_type(AnnotationType.RECTANGLE)
                print("Annotation type set to Rectangle")
            
            elif key == ord('c'):
                self.set_annotation_type(AnnotationType.CIRCLE)
                print("Annotation type set to Circle")
            
            elif key == ord('l'):
                self.set_annotation_type(AnnotationType.LINE)
                print("Annotation type set to Line")
            
            elif key == ord('a'):
                self.set_annotation_type(AnnotationType.ARROW)
                print("Annotation type set to Arrow")
            
            elif key == ord('t'):
                self.set_annotation_type(AnnotationType.TEXT)
                print("Annotation type set to Text")
            
            elif key == ord('p'):
                self.set_annotation_type(AnnotationType.POLYGON)
                print("Annotation type set to Polygon (right-click to finish)")
            
            elif key == ord('f'):
                self.set_annotation_type(AnnotationType.FREEHAND)
                print("Annotation type set to Freehand")
            
            # Check for label change
            elif key >= ord('1') and key <= ord('9'):
                label = chr(key)
                self.set_label(label)
                print(f"Label set to '{label}'")
            
            elif key == ord('0'):
                self.set_label(None)
                print("Label cleared")
            
            # Check for thickness change
            elif key == ord('+') or key == ord('='):
                self.set_thickness(min(10, self.current_thickness + 1))
                print(f"Thickness set to {self.current_thickness}")
            
            elif key == ord('-') or key == ord('_'):
                self.set_thickness(max(1, self.current_thickness - 1))
                print(f"Thickness set to {self.current_thickness}")
            
            # Check for filled toggle
            elif key == ord(' '):
                self.set_filled(not self.current_filled)
                print(f"Filled set to {self.current_filled}")
            
            # Check for delete last annotation
            elif key == 8 or key == 127:  # BACKSPACE or DELETE
                if self.annotations:
                    self.annotations.pop()
                    self.display_image = self.draw_annotations()
                    print("Removed last annotation")
        
        # Clean up
        cv2.destroyWindow('Annotation')
        
        # Return the annotations
        return self.annotations
    
    def save_annotations(self, annotations: List[Annotation], output_path: str) -> None:
        """
        Save annotations to a file.
        
        Args:
            annotations: List of annotations
            output_path: Path to save the annotations
        """
        # Convert annotations to dictionaries
        annotation_dicts = [annotation.to_dict() for annotation in annotations]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(annotation_dicts, f, indent=2)
    
    def load_annotations(self, input_path: str) -> List[Annotation]:
        """
        Load annotations from a file.
        
        Args:
            input_path: Path to the annotations file
            
        Returns:
            List of annotations
        """
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Annotations file not found: {input_path}")
        
        # Load from file
        with open(input_path, 'r') as f:
            annotation_dicts = json.load(f)
        
        # Convert dictionaries to annotations
        annotations = [Annotation.from_dict(data) for data in annotation_dicts]
        
        return annotations
    
    def save_annotated_image(self, image: np.ndarray, annotations: List[Annotation], 
                           output_path: str) -> None:
        """
        Save an image with annotations drawn on it.
        
        Args:
            image: Input image (BGR format)
            annotations: List of annotations
            output_path: Path to save the annotated image
        """
        # Draw annotations on the image
        result = image.copy()
        for annotation in annotations:
            result = annotation.draw(result)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        cv2.imwrite(output_path, result)


# Example usage:
if __name__ == "__main__":
    # Initialize annotator
    annotator = ImageAnnotator()
    
    # Test on a sample image
    sample_image_path = "../data/images/test.jpg"  # Update with your image path
    if os.path.exists(sample_image_path):
        # Load image
        image = cv2.imread(sample_image_path)
        
        if image is not None:
            # Start annotation
            annotations = annotator.annotate_image(image)
            
            # Print annotations
            print(f"Created {len(annotations)} annotations:")
            for i, annotation in enumerate(annotations):
                print(f"  {i+1}. Type: {annotation.type.value}, Points: {annotation.points}, Label: {annotation.label}")
            
            # Save annotations
            if annotations:
                # Save to file
                annotations_path = "../data/annotations/test_annotations.json"
                annotator.save_annotations(annotations, annotations_path)
                print(f"Saved annotations to {annotations_path}")
                
                # Save annotated image
                output_path = "../data/annotations/test_annotated.jpg"
                annotator.save_annotated_image(image, annotations, output_path)
                print(f"Saved annotated image to {output_path}")
                
                # Display the annotated image
                annotated_image = cv2.imread(output_path)
                cv2.imshow("Annotated Image", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Could not load image: {sample_image_path}")
    else:
        print(f"Sample image not found: {sample_image_path}")
