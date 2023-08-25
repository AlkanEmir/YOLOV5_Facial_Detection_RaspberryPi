import torch
import numpy
import cv2
import time

IMG_SIZE = 160
conf_threshold = 0.75
model_path = '/home/pi/Desktop/Facial_Recognition/Models/last_v5.4_r160.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload = True,)

capture = cv2.VideoCapture(0)
start_time = None
while capture.isOpened():
    ret, frame = capture.read()
    
    results = model(frame, size = IMG_SIZE)
    
    bboxes = results.pred[0][:, :4].cpu().numpy()
    confidences = results.pred[0][:, 4].cpu().numpy()
    class_ids = results.pred[0][:, 5].cpu().numpy().astype(int)
    detect = False
    
    for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
        if confidences >= conf_threshold:
            x_min, y_min, x_max, y_max = bbox
            
            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1
            font_thickness = 1
            frame_thickness= 3
            
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, frame_thickness)
            class_name = model.names[class_id]
            label = f'{class_name}: {confidence:.2f}'
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min) + text_size[0], int(y_min) - text_size[1] - 5), color, -1)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 5), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
            
            detect = True
    
    if detect:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= 5:
            print(f'{class_name} detected.') # Change this with light LED for Rpi with breadboard.
            start_time = 0            
    else:
        start_time = None
               
    cv2.imshow('Facial Recognition', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
