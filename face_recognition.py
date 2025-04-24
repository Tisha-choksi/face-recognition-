import cv2
import numpy as np
from PIL import Image
import logging
from .utils import setup_logger

logger = setup_logger()

class FaceDetector:
    def __init__(self):
        """Initialize the FaceDetector class."""
        self.logger = logging.getLogger('face_recognition_app.detector')
        # Load OpenCV's pre-trained face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise RuntimeError("Error: Could not load face cascade classifier")

    def _convert_to_cv2(self, image):
        """
        Convert PIL Image or numpy array to OpenCV format.
        
        Args:
            image: PIL.Image or numpy.ndarray
            
        Returns:
            numpy.ndarray: Image in OpenCV format (BGR)
        """
        try:
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                np_image = np.array(image)
                # Convert RGB to BGR if image is RGB
                if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                    return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                return np_image
            elif isinstance(image, np.ndarray):
                # If already numpy array, ensure it's in BGR format if it's a color image
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            self.logger.error(f"Error converting image format: {str(e)}")
            raise RuntimeError(f"Failed to convert image format: {str(e)}")

    def _convert_to_pil(self, cv2_image):
        """
        Convert OpenCV image to PIL Image format.
        
        Args:
            cv2_image: numpy.ndarray in BGR format
            
        Returns:
            PIL.Image: Converted image
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        except Exception as e:
            self.logger.error(f"Error converting to PIL format: {str(e)}")
            raise RuntimeError(f"Failed to convert to PIL format: {str(e)}")

    def detect_faces(self, image):
        """
        Detect faces in the given image and draw bounding boxes around them.
        
        Args:
            image: PIL.Image or numpy.ndarray
            
        Returns:
            tuple: (processed_image, number_of_faces_detected)
        """
        try:
            if image is None:
                raise ValueError("No image provided")

            self.logger.info("Detecting faces in the image...")
            
            # Convert to OpenCV format
            cv2_image = self._convert_to_cv2(image)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(cv2_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add some padding for text background
                cv2.rectangle(cv2_image, (x, y-25), (x+70, y), (0, 255, 0), -1)
                
                # Add "Face" text
                cv2.putText(
                    cv2_image,
                    'Face',
                    (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

            if len(faces) > 0:
                self.logger.info(f"Successfully detected {len(faces)} faces")
            else:
                self.logger.info("No faces detected in the image")

            # Convert back to PIL Image
            processed_image = self._convert_to_pil(cv2_image)
            return processed_image, len(faces)

        except Exception as e:
            self.logger.error(f"Error during face detection: {str(e)}")
            raise RuntimeError(f"Failed to process image: {str(e)}")

    def compare_faces(self, image1, image2):
        """
        Compare faces between two images and return similarity score.
        
        Args:
            image1: First image for comparison
            image2: Second image for comparison
            
        Returns:
            tuple: (is_match, confidence_score)
        """
        try:
            if image1 is None or image2 is None:
                raise ValueError("Both images must be provided for comparison")

            # For now, just detect if both images contain faces
            _, faces1 = self.detect_faces(image1)
            _, faces2 = self.detect_faces(image2)
            
            # Simple comparison - if both images have faces, consider it a match
            is_match = faces1 > 0 and faces2 > 0
            confidence = 0.8 if is_match else 0.2
            
            return is_match, confidence

        except Exception as e:
            self.logger.error(f"Error during face comparison: {str(e)}")
            raise RuntimeError(f"Failed to compare faces: {str(e)}")
