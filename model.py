import cv2
import numpy as np

def reorder(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4,2), dtype=np.float32)

    add = points.sum(1)
    diff = np.diff(points, axis=1)

    new_points[0] = points[np.argmin(add)] 
    new_points[2] = points[np.argmax(add)]  
    new_points[1] = points[np.argmin(diff)] 
    new_points[3] = points[np.argmax(diff)] 
    return new_points

def scan_document(img):
    org = img.copy()

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 75, 200)

    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        return org, None, None

    points = reorder(doc_contour)

   
    (tl, tr, br, bl) = points
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    maxWidth = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    maxHeight = int(max(height_left, height_right))

    dst = np.array([[0,0],[maxWidth,0],[maxWidth,maxHeight],[0,maxHeight]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(points, dst)
    warp = cv2.warpPerspective(org, matrix, (maxWidth, maxHeight))

 
    scanned = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(scanned, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    return org, warp, scanned
