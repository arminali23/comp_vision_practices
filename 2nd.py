import cv2

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, hierarchy = cv2.findContours(
    edges.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Contours: {len(contours)}")

for c in contours:
    area = cv2.contourArea(c)
    if area < 200:   
        continue

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    sides = len(approx)

    if sides == 3:
        shape_name = "Triangle"
    elif sides == 4:
        aspect_ratio = w / float(h)
        if 0.9 < aspect_ratio < 1.1:
            shape_name = "Square"
        else:
            shape_name = "Rect"
    elif sides == 5:
        shape_name = "Pentagon"
    else:
        shape_name = "Circle"

    cv2.drawContours(orig, [approx], -1, (0, 255, 0), 2)
    cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(orig, shape_name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Edges", edges)
cv2.imshow("Shapes", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()