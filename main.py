import cv2
import dlib
import numpy as np
from PIL import Image

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)
    return landmarks

def apply_face_swap(image1, image2):
    landmarks1 = get_landmarks(image1)
    landmarks2 = get_landmarks(image2)

    if len(landmarks1) == 0 or len(landmarks2) == 0:
        print("No face detected!")
        return image1

    # Assume single face for simplicity
    landmarks1 = landmarks1[0]
    landmarks2 = landmarks2[0]

    # Create a mask from the landmarks and extract face region
    points1 = np.array([(p.x, p.y) for p in landmarks1.parts()])
    points2 = np.array([(p.x, p.y) for p in landmarks2.parts()])

    # Create a convex hull for the face area
    hull1 = cv2.convexHull(points1)
    hull2 = cv2.convexHull(points2)

    # Get the ROI for both images
    rect1 = cv2.boundingRect(hull1)
    rect2 = cv2.boundingRect(hull2)

    # Create masks
    mask1 = np.zeros_like(image1, dtype=np.uint8)
    mask2 = np.zeros_like(image2, dtype=np.uint8)
    
    cv2.fillConvexPoly(mask1, hull1, (255, 255, 255))
    cv2.fillConvexPoly(mask2, hull2, (255, 255, 255))

    # Perform warping
    face1 = image1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]
    face2 = image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]]

    # Resize faces to match the shape of the other face
    face2_resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]))

    # Swap faces (this will need some refinement for real-world use)
    image1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]] = face2_resized

    return image1

# Load your images (ensure both have faces)
image1 = cv2.imread('copy.jpg')  # This would be the target image where Nicolas Cage's face will be swapped in
image2 = cv2.imread('nicolas-cage.jpg')  # This should be an image of Nicolas Cage's face

# Perform face swap
swapped_image = apply_face_swap(image1, image2)

# Convert back to RGB for showing with PIL
swapped_image_rgb = cv2.cvtColor(swapped_image, cv2.COLOR_BGR2RGB)
swapped_image_pil = Image.fromarray(swapped_image_rgb)

cv2.imwrite("swapped_face.jpg", swapped_image)

# Show the swapped image
swapped_image_pil.show()
