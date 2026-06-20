# CLAHE

# Poboljšanje kontrasta.

# clahe = cv2.createCLAHE(
#     clipLimit=2.0,
#     tileGridSize=(8,8)
# )
# Gaussian Blur
# img = cv2.GaussianBlur(img, (3,3), 0)
# Histogram Equalization
# img = cv2.equalizeHist(img)
# Sharpening
# kernel = np.array([
#     [0,-1,0],
#     [-1,5,-1],
#     [0,-1,0]
# ])

# img = cv2.filter2D(img, -1, kernel)