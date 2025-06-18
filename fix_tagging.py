import json
import os
import sys
from src.modules.dataset_creation import DatasetCreator

class JSONSerializableFix:
    """
    Helper class to fix JSON serialization issues in the dataset creator.
    """
    @staticmethod
    def fix_non_serializable(obj):
        """
        Convert non-serializable objects to serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, bool):
            return str(obj)  # Convert boolean to string
        elif isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [JSONSerializableFix.fix_non_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: JSONSerializableFix.fix_non_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)  # Convert other non-serializable objects to string

def fix_dataset_tags(dataset_id):
    """
    Fix JSON serialization issues in dataset tags.
    
    Args:
        dataset_id: ID of the dataset to fix
    """
    # Initialize dataset creator
    creator = DatasetCreator("data/datasets")
    
    try:
        # Get dataset info
        dataset_info = creator.get_dataset_info(dataset_id)
        print(f"Fixing tags for dataset: {dataset_info['name']}")
        
        # Process each image
        for image_id, tags in dataset_info['tags'].items():
            try:
                # Fix automatic tags if present
                if 'automatic_tags' in tags:
                    fixed_tags = JSONSerializableFix.fix_non_serializable(tags['automatic_tags'])
                    tags['automatic_tags'] = fixed_tags
                    
                    # Update tags
                    creator.tag_image(dataset_id, image_id, {'automatic_tags': fixed_tags})
                    print(f"Fixed tags for image {image_id}")
            except Exception as e:
                print(f"Error fixing tags for image {image_id}: {e}")
        
        print(f"Finished fixing tags for dataset {dataset_id}")
        
    except Exception as e:
        print(f"Error fixing dataset tags: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_tagging.py <dataset_id>")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    success = fix_dataset_tags(dataset_id)
    sys.exit(0 if success else 1)
