from flask import Flask, request, jsonify, send_file
import cv2
import dlib
import numpy as np
from PIL import Image
import io

# Initialize Flask app and the dlib face detector and shape predictor
app = Flask(__name__)

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to get landmarks for a face
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)
    return landmarks

# Function to blend faces for smoother transitions
def blend_faces(image1, image2, mask):
    return cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

# Function to apply the face swap
def apply_face_swap(image1, image2):
    landmarks1 = get_landmarks(image1)
    landmarks2 = get_landmarks(image2)

    if len(landmarks1) == 0 or len(landmarks2) == 0:
        return None

    # Assume single face for simplicity
    landmarks1 = landmarks1[0]
    landmarks2 = landmarks2[0]

    # Create a mask from the landmarks and extract face region
    points1 = np.array([(p.x, p.y) for p in landmarks1.parts()])
    points2 = np.array([(p.x, p.y) for p in landmarks2.parts()])

    hull1 = cv2.convexHull(points1)
    hull2 = cv2.convexHull(points2)

    rect1 = cv2.boundingRect(hull1)
    rect2 = cv2.boundingRect(hull2)

    mask1 = np.zeros_like(image1, dtype=np.uint8)
    mask2 = np.zeros_like(image2, dtype=np.uint8)

    cv2.fillConvexPoly(mask1, hull1, (255, 255, 255))
    cv2.fillConvexPoly(mask2, hull2, (255, 255, 255))

    face1 = image1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]
    face2 = image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]]

    face2_resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]))

    image1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]] = face2_resized

    # Blending faces to make it smoother
    blended_image = blend_faces(image1, face2_resized, mask2)
    return blended_image

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, world!'

# Flask route to handle the face swap
@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    # Get the uploaded files (both images)
    file1 = request.files.get('target_image')
    file2 = request.files.get('cage_image')

    if not file1 or not file2:
        return jsonify({'error': 'Both images must be uploaded'}), 400

    # Read the images from the uploaded files
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    img1 = np.array(img1)
    img2 = np.array(img2)

    # Perform the face swap
    swapped_image = apply_face_swap(img1, img2)

    if swapped_image is None:
        return jsonify({'error': 'No faces detected in one or both images'}), 400

    # Convert the swapped image to PIL format for easy returning
    swapped_image_pil = Image.fromarray(cv2.cvtColor(swapped_image, cv2.COLOR_BGR2RGB))

    # Save the image to a BytesIO object for sending back as a response
    img_io = io.BytesIO()
    swapped_image_pil.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='swapped_face.jpg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)