import cv2

img = cv2.imread("test.JPG")


if img is None:
    print('image not found')
    exit()
    
cv2.imshow('original', img)

gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imshow('gray', gray)

blur = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('blur', blur)

edges = cv2.Canny(blur,100,200)
cv2.imshow('edges',edges)

contours, hierarchy = cv2.findContours(
    edges.copy(),           
    cv2.RETR_EXTERNAL,      
    cv2.CHAIN_APPROX_SIMPLE 
)

print(f"Found {len(contours)} contours")
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours", contour_img)

cv2.waitKey(0)
cv2.destroyAllWindows()