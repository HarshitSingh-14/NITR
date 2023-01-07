from sre_constants import SUCCESS
from tkinter.tix import Tree
import cv2
from cv2 import threshold
from djitellopy import tello
import cvzone
from threading import Thread
import time
import keyCapture as key




key.init()


threshold = 0.65
# for detecting duplicates
nmsThreshold =0.2



# cap = cv2.VideoCapture(0)
# cap.set(3,720)
# cap.set(4,480)




NamesOfObjects= []
NameFile = 'coco.names'
ConfigFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WeightFile = 'frozen_inference_graph.pb'
with open(NameFile,'rt') as f:
    NamesOfObjects = f.read().split('\n')
print(NamesOfObjects)


# Using pretrained model   -.... documentation coco
net = cv2.dnn_DetectionModel(WeightFile,ConfigFile)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamoff()
drone.streamon()

# drone.takeoff()


keepRecording = True
img = drone.get_frame_read().frame
height, width, _ = img.shape
video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

while True:
    # success ,img = cap.read()  # for webcam
    img = drone.get_frame_read().frame
    
    classIds, confidence , boundingBox= net.detect(img, confThreshold=threshold, nmsThreshold=nmsThreshold)
    try:
        for classId, conf, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            cvzone.cornerRect(img, box)# draw rectangle
            cv2.putText(img, f'{NamesOfObjects[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass

    # to prevent auto off drone
    drone.send_rc_control(0,0,0,0)
    video.write(img)
    cv2.imshow("img",img)
    

    cv2.waitKey(1)



video.release()
keepRecording = False
# recorder.join()