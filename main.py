import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import signal
from typing import List, Tuple

os.makedirs("./results", exist_ok=True)

def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image and converts it to grayscale.

    Args:
        image_path (str): Path to the image file.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Original image and grayscale image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}. Check the file path.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return img, gray

def compute_harris_response(image: np.ndarray) -> np.ndarray:
    """
    Computes the Harris corner response for an image.
    Reference: https://slazebni.cs.illinois.edu/fall22/assignment3/harris.py

    Args:
        image (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Harris response matrix.
    """
    gx, gy = np.gradient(image)  # Compute gradients
    gauss = cv2.getGaussianKernel(3, 1)
    gauss = gauss * gauss.T  # 2D Gaussian kernel
    
    # Compute structure tensor components
    Ixx = signal.convolve2d(gx**2, gauss, mode='same')
    Iyy = signal.convolve2d(gy**2, gauss, mode='same')
    Ixy = signal.convolve2d(gx * gy, gauss, mode='same')
    
    # Harris response formula 
    # Ref: https://en.wikipedia.org/wiki/Harris_corner_detector 
    return (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-6)

def get_harris_points(harris_resp: np.ndarray, min_distance: int = 10, threshold: float = 0.1) -> List[Tuple[int, int]]:
    """
    Extracts keypoints from the Harris response matrix.
    Reference: https://slazebni.cs.illinois.edu/fall22/assignment3/harris.py

    Args:
        harris_resp (np.ndarray): Harris response matrix.
        min_distance (int): Minimum spacing between detected points.
        threshold (float): Response threshold for selecting keypoints.
    
    Returns:
        List[Tuple[int, int]]: List of detected keypoints (x, y).
    """
    corner_thresh = max(harris_resp.ravel()) * threshold
    coords = np.argwhere(harris_resp > corner_thresh)
    values = harris_resp[harris_resp > corner_thresh]
    index = np.argsort(values)[::-1]  # Sort in descending order

    allowed = np.zeros(harris_resp.shape, dtype=bool)
    allowed[min_distance:-min_distance, min_distance:-min_distance] = True

    filtered_coords = []
    for i in index:
        y, x = coords[i]
        if allowed[y, x]:
            filtered_coords.append((x, y))
            allowed[y - min_distance:y + min_distance, x - min_distance:x + min_distance] = False

    return filtered_coords

def extract_descriptors(image: np.ndarray, keypoints: List[Tuple[int, int]], size: int = 9) -> np.ndarray:
    """
    Extracts normalized patch descriptors for keypoints.

    Args:
        image (np.ndarray): Grayscale image.
        keypoints (List[Tuple[int, int]]): Detected keypoints.
        size (int): Size of the descriptor patch.
    
    Returns:
        np.ndarray: Array of descriptors.
    """
    descriptors = []
    half_size = size // 2
    padded = np.pad(image, half_size, mode='constant')

    for x, y in keypoints:
        patch = padded[y:y + size, x:x + size].flatten()
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
        descriptors.append(patch)

    return np.array(descriptors)

def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> List[Tuple[int, int]]:
    """
    Matches descriptors using the ratio test.

    Args:
        desc1 (np.ndarray): Descriptors from image 1.
        desc2 (np.ndarray): Descriptors from image 2.
        ratio (float): Lowe's ratio test threshold.
    
    Returns:
        List[Tuple[int, int]]: List of matched descriptor indices.
    """
    distances = cdist(desc1, desc2, 'sqeuclidean')
    matches = []
    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])
        best, second_best = distances[i, sorted_indices[:2]]
        if best < ratio * second_best:
            matches.append((i, sorted_indices[0]))
    return matches

def estimate_homography_ransac(kp1: List[Tuple[int, int]], kp2: List[Tuple[int, int]], 
                                matches: List[Tuple[int, int]], ransac_thresh: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the homography matrix using RANSAC.
    Reference: https://github.com/dastratakos/Homography-Estimation/blob/main/imageAnalysis.py
    Args:
        kp1 (List[Tuple[int, int]]): Keypoints from image 1.
        kp2 (List[Tuple[int, int]]): Keypoints from image 2.
        matches (List[Tuple[int, int]]): Matched keypoints.
        ransac_thresh (float): RANSAC threshold.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Homography matrix and mask.
    """
    src_pts = np.float32([kp1[i] for i, _ in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[j] for _, j in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, mask

def warp_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Warps and stitches two images using a homography matrix.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        H (np.ndarray): Homography matrix.
    
    Returns:
        np.ndarray: Stitched panorama.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    all_corners = np.vstack((warped_corners, corners))
    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    result = cv2.warpPerspective(img1, np.dot(translation, H), (x_max - x_min, y_max - y_min))
    result[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

    return result


def stitch_images(img_path1: str, img_path2: str) -> None:
    """
    Main function to stitch two images together using feature matching and homography estimation.
    """

    # Got Error for Threshold 16.0 and Ratio > 0.75, Rest the results are in ./results folder
    
    RANSAC_THRESHOLDS = [2.0, 4.0, 8.0, 10.0, 16.0]
    RATIOS = [0.45, 0.65, 0.75, 0.8, 0.95]

    
    for ransac_threshold in RANSAC_THRESHOLDS:
        for ratio in RATIOS:

            try:
                img1, gray1 = load_and_preprocess(img_path1)
                img2, gray2 = load_and_preprocess(img_path2)
                
                harris1 = compute_harris_response(gray1)
                harris2 = compute_harris_response(gray2)
                
                keypoints1 = get_harris_points(harris1, min_distance=5, threshold=0.05)
                keypoints2 = get_harris_points(harris2, min_distance=5, threshold=0.05)
                
                desc1 = extract_descriptors(gray1, keypoints1)
                desc2 = extract_descriptors(gray2, keypoints2)
                
                matches = match_descriptors(desc1, desc2, ratio=ratio)

                # Compare Parameters
                print(f"ransac_threshold: {ransac_threshold} and ratio: {ratio}")
                print(f"Number of matches: {len(matches)}")
                print("-"*100)

                if len(matches) < 4:
                    print(f"Not enough matches found! for ransac_threshold: {ransac_threshold} and ratio: {ratio}")
                    continue

                H, _ = estimate_homography_ransac(keypoints1, keypoints2, matches, ransac_threshold)
                panorama = warp_images(img1, img2, H)
                
                plt.imsave(f"./results/panorama_ransac_{ransac_threshold}_ratio_{ratio}.png", cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))

            except Exception as e:
                print(f"Error stitching images for ransac_threshold: {ransac_threshold} and ratio: {ratio}")
                print(e)


if __name__ == "__main__":
    stitch_images('Image_1.jpg', 'Image_2.jpg')
