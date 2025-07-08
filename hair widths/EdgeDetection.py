import cv2
import numpy as np

def nothing(x):
    pass

def edge_detection_gui(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return

    cv2.namedWindow('Canny Edge Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny Edge Detector', 800, 400)

    # Create trackbars
    cv2.createTrackbar('Lower', 'Canny Edge Detector', 50, 500, nothing)
    cv2.createTrackbar('Upper', 'Canny Edge Detector', 150, 500, nothing)
    cv2.createTrackbar('Canny On/Off', 'Canny Edge Detector', 1, 1, nothing)

    while True:
        lower = cv2.getTrackbarPos('Lower', 'Canny Edge Detector')
        upper = cv2.getTrackbarPos('Upper', 'Canny Edge Detector')
        toggle = cv2.getTrackbarPos('Canny On/Off', 'Canny Edge Detector')

        if toggle == 1:
            edges = cv2.Canny(img, lower, upper)
            display = np.hstack((img, edges))
        else:
            display = np.hstack((img, img))

        cv2.imshow('Canny Edge Detector', display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

def get_hair_contours(image_path, lower=50, upper=150):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return []
    edges = cv2.Canny(img, lower, upper)
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Example usage
edge_detection_gui("hair widths/data/1.jpg")  # Replace with your image path
