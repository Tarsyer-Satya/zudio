import cv2
import numpy as np

def predict_people(img,model):
    results = model.predict(img, verbose = False)
    count = 0
    indexes = []
    for ind,i in enumerate(results[0].boxes.cls):
        if i == 0:
            indexes.append(ind)
    results[0].boxes.xyxy
    l = []
    for ind,i in enumerate(results[0].boxes.xyxy.numpy()):
        if ind in indexes:
            l.append(i)
    return l

def draw(bb,img):
    image = img.copy()
    vertices1 = np.array(bb, np.int32).reshape((-1, 1, 2))
    return cv2.polylines(image, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)


def crop_roi(bb, img):
    # drawn_img = draw(bb,img)
    x1,y1,x2,y2 = bb
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def find_centroids(boxes):
    centroids = []
    for l in boxes:
        centroid_x = (l[0] + l[2])//2
        centroid_y = (l[1] + l[3])//2
        centroids.append((centroid_x, centroid_y))
    return centroids
        

def check(point, roi):
    return point_inside_polygon(point, roi)

        

def plot_people(l,img):
    image = img.copy()
    for i in l:
        start_point = (int(i[0]), int(i[1])) 
        end_point = (int(i[2]), int(i[3])) 
        color = (255, 0, 0) 
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


from shapely.geometry import Point, Polygon

def point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    Args:
        point (tuple): Tuple containing the coordinates of the point (x, y).
        polygon (list of tuples): List of tuples containing the coordinates of the polygon vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    point = Point(point)
    polygon = Polygon(polygon)
    return polygon.contains(point)


