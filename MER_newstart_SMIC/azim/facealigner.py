from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# aligned_face = DeepFace.detectFace("img.jpg")
# print(type(aligned_face))
# plt.imshow(aligned_face)
# plt.imsave('test.png', aligned_face)

img = cv2.imread("img.jpg")
img_raw = img.copy()
