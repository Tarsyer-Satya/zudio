from ultralytics import YOLO
import cv2
# import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import time
# import cairo
from helper_function_person_counting import *
from datetime import datetime
import os
import json
import numpy as np
# parser = argparse.ArgumentParser(description='A script to process an image')
# parser.add_argument('image', type=str, help='Path to the image file')
# args = parser.parse_args()

store_code = 'z431'
store_name = 'Zudio-Aeromall'
camera_no = 3


bb1 = [[515,177],[580, 4], [1279, 243], [1279, 396]]
bb2 = [[714,46],[1006,3],[1279,129],[1279,240]]

image_path = '/tmp/image.jpg'

image_path = 'fc_road.jpg'

# ALERT_IMAGE_PATH = '/tmp/alert_images/'
# os.system(f'mkdir -p {ALERT_IMAGE_PATH}')

# json_path = '/tmp/json_data/'
# os.system(f'mkdir -p {json_path}')


model = YOLO('yolov8n.pt')  

time_interval = 30

last_executed_time = time.monotonic()-time_interval


while True:


    # run below part after every x seconds

    if abs(time.monotonic()-last_executed_time) > time_interval:
        last_executed_time = time.monotonic()
            
        start_of_code = time.monotonic()

        img = cv2.imread(image_path)
            
        person_boxes = predict_people(img,model)
        person_boxes_centroid = find_centroids(person_boxes) 

        # plot_image = plot_people(person_boxes,img)
        # plt.imshow(plot_image)
        # plt.show()

        bb1 = [(515,177),(580, 4), (1279, 243), (1279, 396)]
        bb2 = [(714,46),(1006,3),(1279,129),(1279,240)]
        roi = draw(bb1,img)
        roi = draw(bb2,roi)
        # plt.imshow(roi)

        count1,count2 = 0,0
        for i in range(len(person_boxes)):
            if check(person_boxes_centroid[i], bb1):
                count1 += 1
            if check(person_boxes_centroid[i], bb2):
                count2 += 1

        

        cv2.rectangle(roi, (1090,20), (1190,100), (0,255,0) , -1)
        cv2.putText(roi, "L1-"+str(count1), (1100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.putText(roi, "L2-"+str(count2), (1100,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        
        #plt.imshow(roi)
        #plt.show()

        main_image = cv2.resize(roi, (640, 480))

        current_datetime =  datetime.now().strftime("%d-%m-%y_%H-%M-%S")    
        filename_format = f'{current_datetime}_person-count-2nd-lane'
        json_name_format = f'{filename_format}.json'
        image_name_format = f'{filename_format}.jpg'

        if count1 >= 2 and count2 >= 2:
            cv2.imwrite(f'{ALERT_IMAGE_PATH}{image_name_format}', main_image)

        data = {
            'store_name': store_name,
            'store_code': store_code,
            'camera_no': camera_no,
            'date_time': current_datetime,
            'L1_count': count1,
            'L2_count' : count2,
            'count': count1 + count2,
        }

        # Convert the data to JSON format
        # json_data = json.dumps(data, indent=4)

        # Create the full path for the JSON file
        # json_file_path = os.path.join(json_path, json_name_format)

        # Write the JSON data to the specified path
        # with open(json_file_path, 'w') as json_file:
        #     json_file.write(json_data)

        # print(f'Saved to {json_file_path}, count = {count1 + count2}')

        # print(person_boxes)
        print("Time took in seconds: ", time.monotonic()-start_of_code)

        cv2.imshow("Image", main_image)
        # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# bye bye
cv2.destroyAllWindows()
