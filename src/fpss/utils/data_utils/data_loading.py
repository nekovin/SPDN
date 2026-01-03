import glob
import os
import numpy as np
from skimage import io
import re

def load_patient_data(base_path, verbose=False):
    """Load patient OCT scan data from the specified directory.
    
    Args:
        base_path (str): Path to directory containing image files
        verbose (bool): Whether to print loading information
        
    Returns:
        list: List of loaded and normalized OCT scan images
    """
    all_files = []
    for ext in ["*.tiff", "*.tif", "*.png", "*.jpg"]:
        all_files.extend(glob.glob(os.path.join(base_path, ext)))
    
    if not all_files:
        if verbose:
            print("No image files found")
        return []
    
    # Define the sorting function
    def extract_number_key(filename):
        # Extract the number inside brackets using regex
        match = re.search(r'\((\d+)\)', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0  # Default value if no number is found
    
    # Sort files
    files = sorted(all_files, key=extract_number_key)
    
    if verbose:
        print(f"Found {len(files)} files")

    oct_scans = []
    for file in files:
        try:
            img = io.imread(file)
            
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
                
            oct_scans.append(img)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return oct_scans