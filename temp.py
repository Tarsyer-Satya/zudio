import cv2
from helper_function_person_counting import *
import numpy as np


image = cv2.imread('fc_road.jpg')



bb1 = [[515,177],[580, 4], [1279, 243], [1279, 396]]
bb2 = [[714,46],[1006,3],[1279,129],[1279,240]]


vertices1 = np.array(bb1, np.int32).reshape((-1, 1, 2))
cv2.polylines(image, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)

vertices2 = np.array(bb2, np.int32).reshape((-1, 1, 2))
cv2.polylines(image, [vertices2], isClosed=True, color=(0, 0, 255), thickness=2)


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# polygon_vertices = [(0, 0), (0, 5), (5, 5), (5, 0)]  # Example polygon vertices
# point_to_check = (10, 2)  # Example point to check

# print(point_inside_polygon(point_to_check, polygon_vertices))  # Output: True
