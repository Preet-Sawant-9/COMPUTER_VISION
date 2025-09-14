import cv2
import numpy as np

# STEP 2 – READ THE IMAGE (update the filename to your image if needed)
IMAGE_PATH = "coins.png"  # Replace this with your image filename if different
orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Cannot open {IMAGE_PATH}")

# STEP 3 – CONVERT TO GRAYSCALE
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# STEP 4 – APPLY GAUSSIAN BLUR TO REDUCE NOISE
blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)

# STEP 5 – EDGE DETECTION USING CANNY
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# STEP 6 – FIND CONTOURS AND OUTLINE OBJECTS (COINS)
output = orig.copy()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
object_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100:  # Filter out spurious small objects (e.g., noise)
        continue
    object_count += 1
    # Draw thick red outlines around detected objects
    cv2.drawContours(output, [cnt], 0, (0, 0, 255), 6)

print(f"Total number of objects detected: {object_count}")

# ----- HOUGH CIRCLE TRANSFORM WITH TRACKBARS FOR PARAMETER TUNING -----

def nothing(x):
    pass  # Dummy function for creating trackbars

# Create a window for interactive circle detection
cv2.namedWindow("Detected Circles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Circles", 800, 800)

# Add sliders for all common Hough Circle parameters
cv2.createTrackbar("minRadius", "Detected Circles", 10, 100, nothing)
cv2.createTrackbar("maxRadius", "Detected Circles", 80, 200, nothing)
cv2.createTrackbar("param1", "Detected Circles", 100, 300, nothing)
cv2.createTrackbar("param2", "Detected Circles", 30, 100, nothing)
cv2.createTrackbar("minDist", "Detected Circles", 60, 200, nothing)

while True:
    # Read slider values for parameters
    minRadius = cv2.getTrackbarPos("minRadius", "Detected Circles")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Detected Circles")
    param1 = cv2.getTrackbarPos("param1", "Detected Circles")
    param2 = cv2.getTrackbarPos("param2", "Detected Circles")
    minDist = cv2.getTrackbarPos("minDist", "Detected Circles")

    # Ensure minimum radius does not exceed maximum
    if minRadius >= maxRadius:
        maxRadius = minRadius + 1

    temp_output = output.copy()
    # Use Hough Circles to find circles in the image with current slider values
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, minDist),
        param1=max(1, param1),
        param2=max(1, param2),
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for x, y, r in circles:
            # Draw thin blue circle outline
            cv2.circle(temp_output, (x, y), r, (0, 255, 0), 2)
            # Draw small filled red center
            cv2.circle(temp_output, (x, y), 6, (0, 0, 255), -1)
        cv2.setWindowTitle("Detected Circles", f"Detected Circles - Total: {len(circles)}")
    else:
        cv2.setWindowTitle("Detected Circles", "Detected Circles - Total: 0")

    cv2.imshow("Detected Circles", temp_output)
    key = cv2.waitKey(50) & 0xFF
    if key == 27 or key == ord('q'):  # Exit on 'q' or ESC
        break

cv2.destroyAllWindows()






#with gui radius




import cv2
import numpy as np
# ---------------------------------------
# Trackbar Callback (required for createTrackbar)
def nothing(x):
    pass
# ---------------------------------------
# Shape Detection by Approximation
def detect_shape(approx, contour):
    sides = len(approx)
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]
                v1 = p1 - p2
                v2 = p3 - p2
                angle = np.degrees(
                    np.arccos(np.dot(v1, v2) /
                             (np.linalg.norm(v1) * np.linalg.norm(v2)))
                )
                angles.append(angle)
            if np.allclose(angles, 90, atol=10):
                return "Rectangle"
            elif np.isclose(angles[0], angles[2], atol=10):
                return "Rhombus"
            else:
                return "Trapezoid"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif 8 <= sides <= 10:
        return "Star"
    elif sides > 10:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "Unknown"
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.8:
            return "Circle"
        return "Star Polygon"
    else:
        return "Polygon"
# ---------------------------------------
# Load and Preprocess Image
image = cv2.imread("shapes.png")  # Ensure this image exists in your folder
resized = cv2.resize(image, (600, 600))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
# ---------------------------------------
# Create Fullscreen Window and Trackbars
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.createTrackbar("Hough Param1", "Result", 100, 300, nothing)
cv2.createTrackbar("Hough Param2", "Result", 30, 100, nothing)
cv2.createTrackbar("Min Radius", "Result", 10, 100, nothing)
cv2.createTrackbar("Max Radius", "Result", 100, 200, nothing)
while True:
    output = resized.copy()
    total_shapes = 0
    # -----------------------------------
    # Circle Detection (Hough Transform parameters from trackbars)
    p1 = cv2.getTrackbarPos("Hough Param1", "Result")
    p2 = cv2.getTrackbarPos("Hough Param2", "Result")
    minR = cv2.getTrackbarPos("Min Radius", "Result")
    maxR = cv2.getTrackbarPos("Max Radius", "Result")
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
    circle_centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0, :]:
            circle_centers.append((cx, cy))
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(output, "Circle", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            total_shapes += 1
    # -----------------------------------
    # Contour Detection and Shape Approximation
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Avoid double-counting if near a detected circle
            if any(np.linalg.norm(np.array([cx, cy]) - np.array(center)) < 20 for center in circle_centers):
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            shape = detect_shape(approx, cnt)
            cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)
            cv2.putText(output, shape, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            total_shapes += 1
    # -----------------------------------
    # Show Shape Count
    cv2.putText(output, f"Total Shapes: {total_shapes}", (10, 590),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # -----------------------------------
    # Display Result
    cv2.imshow("Result", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
