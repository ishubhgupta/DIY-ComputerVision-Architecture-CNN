import cv2
import os

def process_image(image_path):
    return cv2.imread(image_path)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    return output_path

def apply_grayscale(image_path):
    image = process_image(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path = os.path.join('static', 'uploads', 'grayscale_' + os.path.basename(image_path))
    return save_image(gray_image, output_path)

def apply_blur(image_path, blur_value):
    image = process_image(image_path)
    blurred_image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
    output_path = os.path.join('static', 'uploads', 'blur_' + os.path.basename(image_path))
    return save_image(blurred_image, output_path)

def apply_edge_detection(image_path):
    image = process_image(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    output_path = os.path.join('static', 'uploads', 'edges_' + os.path.basename(image_path))
    return save_image(edges, output_path)
