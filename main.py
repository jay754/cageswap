from flask import Flask, request, jsonify, send_file
import cv2
import dlib
import numpy as np
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)

# Configuration
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
DEBUG_OUTPUT = False  # Set to True to save debug images

# Check if shape predictor file exists
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"Warning: {SHAPE_PREDICTOR_PATH} not found. Face detection will fail.")

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    predictor = None

def get_landmarks(image):
    """Extract facial landmarks from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if DEBUG_OUTPUT:
        print(f"Detected {len(faces)} faces in the image.")
    
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)
    return landmarks

def create_face_mask(image, landmarks):
    """Create a mask for the face region based on landmarks."""
    points = np.array([(p.x, p.y) for p in landmarks.parts()])
    hull = cv2.convexHull(points)
    
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    return mask, hull

def swap_faces(target_image, source_image):
    """Swap faces between two images."""
    if predictor is None:
        return None, "Shape predictor not loaded"
    
    # Convert images to BGR if they're in RGB format
    if len(target_image.shape) == 3 and target_image.shape[2] == 3:
        target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    else:
        target_bgr = target_image
        
    if len(source_image.shape) == 3 and source_image.shape[2] == 3:
        source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    else:
        source_bgr = source_image
    
    # Get facial landmarks
    target_landmarks = get_landmarks(target_bgr)
    source_landmarks = get_landmarks(source_bgr)
    
    # Check if faces were detected
    if not target_landmarks:
        return None, "No face detected in target image"
    if not source_landmarks:
        return None, "No face detected in source image"
    
    # Use the first face from each image
    target_face_landmarks = target_landmarks[0]
    source_face_landmarks = source_landmarks[0]
    
    # Create masks for both faces
    target_mask, target_hull = create_face_mask(target_bgr, target_face_landmarks)
    source_mask, source_hull = create_face_mask(source_bgr, source_face_landmarks)
    
    # Get bounding rectangles
    target_rect = cv2.boundingRect(target_hull)
    source_rect = cv2.boundingRect(source_hull)
    
    # Extract face regions
    target_face = target_bgr[target_rect[1]:target_rect[1]+target_rect[3], 
                         target_rect[0]:target_rect[0]+target_rect[2]]
    source_face = source_bgr[source_rect[1]:source_rect[1]+source_rect[3], 
                         source_rect[0]:source_rect[0]+source_rect[2]]
    
    # Resize source face to match target face dimensions
    source_face_resized = cv2.resize(source_face, (target_rect[2], target_rect[3]))
    
    # Create a copy of the target image
    result_image = target_bgr.copy()
    
    # Replace the face region in the target image with the resized source face
    result_image[target_rect[1]:target_rect[1]+target_rect[3], 
                target_rect[0]:target_rect[0]+target_rect[2]] = source_face_resized
    
    # Apply seamless cloning for better blending
    center = (target_rect[0] + target_rect[2] // 2, 
              target_rect[1] + target_rect[3] // 2)
    
    try:
        # Use seamless cloning for better blending
        result_image = cv2.seamlessClone(
            source_face_resized, 
            target_bgr, 
            target_mask[target_rect[1]:target_rect[1]+target_rect[3], 
                        target_rect[0]:target_rect[0]+target_rect[2]], 
            center, 
            cv2.NORMAL_CLONE
        )
    except cv2.error as e:
        # If seamless cloning fails, continue with the basic swap
        print(f"Seamless cloning failed: {e}")
    
    # Convert back to RGB for return
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Save debug images if enabled
    if DEBUG_OUTPUT:
        cv2.imwrite('debug_target_face.jpg', target_face)
        cv2.imwrite('debug_source_face.jpg', source_face)
        cv2.imwrite('debug_source_resized.jpg', source_face_resized)
        cv2.imwrite('debug_result.jpg', result_image)
    
    return result_rgb, "Face swap successful"

@app.route('/hello', methods=['GET'])
def hello():
    """Test endpoint to verify the server is running."""
    return 'Hello, world!'

@app.route('/swap_faces', methods=['POST'])
def handle_face_swap():
    """API endpoint to swap faces between two images."""
    # Get the uploaded files
    target_file = request.files.get('target_image')
    source_file = request.files.get('cage_image')  # Keeping original name for compatibility
    
    if not target_file or not source_file:
        return jsonify({'error': 'Both target_image and cage_image must be uploaded'}), 400
    
    try:
        # Read images
        target_img = np.array(Image.open(target_file))
        source_img = np.array(Image.open(source_file))
        
        # Perform face swap
        result_img, message = swap_faces(target_img, source_img)
        
        if result_img is None:
            return jsonify({'error': message}), 400
        
        # Convert to PIL Image and save to BytesIO
        output_img = Image.fromarray(result_img)
        img_io = io.BytesIO()
        output_img.save(img_io, 'JPEG')
        img_io.seek(0)
        
        # Return the swapped image
        return send_file(
            img_io, 
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='swapped_face.jpg'
        )
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)