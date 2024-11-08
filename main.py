import cv2
import os
import requests
import time
import replicate
from PIL import Image

# Configurations
IMAGE_DIR = './photos'  # Directory containing carousel images
CAROUSEL_INTERVAL = 5  # Seconds between switching images
MARGIN_RATIO = 0.2 # How much space aroudn the face to capture
FACE_SWAP_MODEL = "xiankgx/face-swap:cff87316e31787df12002c9e20a78a017a36cb31fde9862d8dedd15ab29b7288" #face swap model on replicate

def load_images_from_directory(directory):
    """Load images from a given directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            images.append(cv2.imread(img_path))
    return images

def initialize_webcam():
    """Initialize webcam capture."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to initialize webcam")
    return cap

def detect_face(frame):
    """Detects a face in a given frame using OpenCV's Haar cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Calculate the margin to add around the face
        margin_x = int(w * MARGIN_RATIO)
        margin_y = int(h * MARGIN_RATIO)

        # Adjust the rectangle to include the margin, ensuring it stays within the frame boundaries
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(frame.shape[1], x + w + margin_x)
        y_end = min(frame.shape[0], y + h + margin_y)

        # Crop the frame with the margin
        return frame[y_start:y_end, x_start:x_end]
    return None

def save_temp_image(image, filename):
    """Saves a temporary image to disk."""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img.save(filename)
    return filename

def face_swap(base_image, face_image):
    """Calls the Replicate API to perform face-swapping."""
    base_image_path = save_temp_image(base_image, 'base_image.jpg')
    face_image_path = save_temp_image(face_image, 'face_image.jpg')

    input_data = {
        "local_source": open(face_image_path, "rb"),
        "local_target": open(base_image_path, "rb")
    }

    response = replicate.run(
        FACE_SWAP_MODEL,
        input=input_data
    )

    if response and "image" in response:
        image_url = response["image"].url
        swapped_image_path = 'swapped_image.png' 

        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(swapped_image_path, 'wb') as f:
                f.write(image_response.content)

        # Load the saved image using OpenCV
        swapped_image = cv2.imread(swapped_image_path)
        return swapped_image


    print("Face swap failed.")
    return base_image  # Return the base image if face swap fails

def display_image(base_image, face_image):
    """Displays either the base image or a face-swapped version."""
    if face_image is not None:
        base_image = face_swap(base_image, face_image)
    cv2.imshow('Carousel', base_image)
    cv2.waitKey(1)  # Small delay for image rendering

def main():
    images = load_images_from_directory(IMAGE_DIR)
    if not images:
        print("No images found in the directory.")
        return

    cap = initialize_webcam()
    i = 0
    n = len(images)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            face = detect_face(frame)
            display_image(images[i % n], face)
            i += 1
            time.sleep(CAROUSEL_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting program.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()