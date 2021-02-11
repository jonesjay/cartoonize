import cv2        #Introduces OpenCV library
import numpy as np

#For a cartoon effect we need two things: edge and color palette


img = cv2.imread('screenshot.png') 
cv2.imshow('Initial image', img)


def color_quantization(img, k):
    #Transform the image
    data = np.float32(img).reshape((-1, 3))

    #determine the criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    #Implement K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


# In this we detect the edge of the image using cv2.adptiveThreshold()
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Change the image to grayscale 
    gray_blur = cv2.medianBlur(gray, blur_value)  # Reduce the noise of blurred grayscale image using cv2.medianBlur
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)    
    return edges


# Larger blur value means fewer blacknoises appear. Larger line value means a more emphasized image
line_size = 7
blur_value = 7
edges = edge_mask(img, line_size, blur_value)
cv2.imshow("Edge_mask", edges)
    
# A drawing has less color than a photo. Use color quantization to reduce number of colors.
# We adjust the value of total colors to 9.The K-Means clustering algorithm in OpenCV is applied.
total_color = 9
    
img = color_quantization(img,total_color)
cv2.imshow("Color_quantization", img)

# Reduce noise in the image using a bilateral filter. d- diameter of each neighbourhood pixel
# sigmaColor - larger value meeans larger areas of semi-equal color
# sigmaSpace - larger value means that farther pixels will influence each other as long as their colors are close enough
blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
cv2.imshow("Blurred_img", blurred)


# Combine the edge mask and the color processed image with cv2.bitwise_and
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
cv2.imshow("Combined_img", cartoon)
cv2.waitKey(0)
cv2.destroyWindow("Combined_img") 


