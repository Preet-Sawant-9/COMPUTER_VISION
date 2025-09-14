#part 1 


import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('building.jpg')
h, w, c = img.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([img.reshape((-1, 3)), X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
kmeans = KMeans(n_clusters=5).fit(features)
labels = kmeans.labels_.reshape((h, w))
cv2.imshow('Segmented with XY', labels.astype('uint8') * 50)
cv2.waitKey(0)
cv2.destroyAllWindows()



#part 2 
import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('building.jpg')
h, w, c = img.shape

# Color-only clustering (no spatial coordinates)
features_color = img.reshape((-1, 3))
kmeans_color = KMeans(n_clusters=5, random_state=0).fit(features_color)
labels_color = kmeans_color.labels_.reshape((h, w))

cv2.imshow('Segmented (Color Only)', labels_color.astype('uint8') * 50)
cv2.waitKey(0)
cv2.destroyAllWindows()

#part 3 
import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('building.jpg')
h, w, c = img.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([img.reshape((-1, 3)), X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

for n_clusters in [2, 3, 7, 10]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_.reshape((h, w))
    cv2.imshow(f'Segmented (Clusters={n_clusters})', labels.astype('uint8') * int(255/(n_clusters-1)))
    cv2.waitKey(0)

cv2.destroyAllWindows()


#part 4 
import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('building.jpg')
h, w, c = img.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([img.reshape((-1, 3)), X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
kmeans = KMeans(n_clusters=5).fit(features)
labels = kmeans.labels_.reshape((h, w))

# Visualize segmentation
cv2.imshow('Segmented with XY', labels.astype('uint8') * 50)

# Print cluster centers
print("Cluster centers (B, G, R, X, Y):")
for idx, center in enumerate(kmeans.cluster_centers_):
    b, g, r, x, y = center
    print(f"Cluster {idx+1}: B={b:.1f}, G={g:.1f}, R={r:.1f}, X={x:.1f}, Y={y:.1f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
