import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('building.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Reshape into 2D array of pixels
data = img_rgb.reshape((-1, 3))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
labels = kmeans.fit_predict(data)

# Replace pixel values with their cluster center colors
segmented_img = kmeans.cluster_centers_[labels]
segmented_img = segmented_img.reshape(img_rgb.shape).astype(np.uint8)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Segmented", cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

# Plot KMeans clustering in RGB color space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sample = data[:100]  # Plot a sample for speed
sample_labels = labels[:100]
ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=sample_labels, cmap='tab10')
ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue')
ax.set_title('KMeans Clustering in RGB Color Space')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()





# part 2
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('building.jpg')
img_bgr = img.copy()  # Already in BGR

# Reshape into 2D array of pixels
data = img_bgr.reshape((-1, 3))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
labels = kmeans.fit_predict(data)

# Replace pixel values with their cluster center colors
segmented_img = kmeans.cluster_centers_[labels]
segmented_img = segmented_img.reshape(img_bgr.shape).astype(np.uint8)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Segmented", segmented_img)

# Plot KMeans clustering in BGR color space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sample = data[:100]  # Plot a sample for speed
sample_labels = labels[:100]
ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=sample_labels, cmap='tab10')
ax.set_xlabel('Blue')
ax.set_ylabel('Green')
ax.set_zlabel('Red')
ax.set_title('KMeans Clustering in BGR Color Space')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



#part 3 
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('building.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reshape into 2D array of pixels (single channel)
data = gray.reshape((-1, 1))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
labels = kmeans.fit_predict(data)

# Replace pixel values with their cluster center values
segmented_img = kmeans.cluster_centers_[labels]
segmented_img = segmented_img.reshape(gray.shape).astype(np.uint8)

# Show results
cv2.imshow("Original Gray", gray)
cv2.imshow("Segmented Gray", segmented_img)

# Plot KMeans clustering in grayscale
plt.figure(figsize=(8, 4))
sample = data[:100].flatten()         # Sample for speed
sample_labels = labels[:100]
plt.scatter(np.arange(len(sample)), sample, c=sample_labels, cmap='tab10', s=10)
plt.xlabel('Pixel Index (sampled)')
plt.ylabel('Gray Value')
plt.title('KMeans Clustering in Grayscale')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
