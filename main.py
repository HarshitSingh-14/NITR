from sre_constants import SUCCESS
from tkinter.tix import Tree
import cv2
from cv2 import threshold
from djitellopy import tello
import cvzone
from threading import Thread
from time import sleep
import time
import keyCapture as key



key.init()
drone = tello.Tello()
drone.connect()
global image

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

net = cv2.dnn_DetectionModel(WeightFile,ConfigFile)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


print(drone.get_battery())
drone.streamoff()
drone.streamon()



def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 33
    if key.getKey("LEFT"):
        lr = -speed*2
    elif key.getKey("RIGHT"):
        lr = speed*2


    if key.getKey("UP"):
        fb = speed*2
    elif key.getKey("DOWN"):
        fb = -speed*2





# Boost mode
    if key.getKey("LEFT") and key.getKey("LSHIFT"):
        lr = -speed*3
    elif key.getKey("RIGHT") and key.getKey("LSHIFT"):
        lr = speed*3

    if key.getKey("UP") and key.getKey("LSHIFT"):
        fb = speed*3
    elif key.getKey("DOWN") and key.getKey("LSHIFT"):
        fb = -speed*3

    





    if key.getKey("w"):
        ud = speed
    elif key.getKey("s"):
        ud = -speed


    if key.getKey("a"):
        yv = -(speed)
    elif key.getKey("d"):
        yv = speed


# Boost mode

    if key.getKey("w")  and key.getKey("LSHIFT"):
        ud = speed*2
    elif key.getKey("s")  and key.getKey("LSHIFT"):
        ud = -speed*2
    if key.getKey("a")  and key.getKey("LSHIFT"):
        yv = -(speed)*2
    elif key.getKey("d")  and key.getKey("LSHIFT"):
        yv = speed*2




#quit process and land
    if key.getKey("q"):
        drone.land()
        sleep(3)

#battery
    if key.getKey("b"):
        print(drone.get_battery())




# photo keech
    if key.getKey("p"):
        cv2.imwrite(f'myFiles/Img/{time.time()}.jpg', img)
        # to restrict img
        time.sleep(0.4)


# takeoff 
    if key.getKey("t"):
        drone.takeoff()
    return [lr, fb, ud, yv]

keepRecording = True
img = drone.get_frame_read().frame

while True:
    # success ,img = cap.read()  # for webcam
    img = drone.get_frame_read().frame
    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    classIds, confidence , boundingBox= net.detect(img, confThreshold=threshold, nmsThreshold=nmsThreshold)
    try:
        for classId, conf, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            cvzone.cornerRect(img, box)# draw rectangle
            # if(classId==26 or classId==27 or classId==28 or classId==29 or classId==37 or classId==44 or classId==45 or classId==46 or classId ==47 or classId ==48  or classId==49 or classId ==50 or classId==51):
            #     cv2.putText(img, f' [** Non Recyclable Wastes **]-> -->{round(conf * 100, 2)}',
            #                 (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 1, (0, 255, 0), 2)
            # elif(classId==52 or classId==53 or classId==54 or classId ==55 or classId ==56  or classId==57 or classId ==58 or classId==59 or classId==60 ):
            #     cv2.putText(img, f' [** Recyclable Compose Wastes **] -> -->{round(conf * 100, 2)}',
            #                 (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 1, (0, 255, 0), 2)
            # else:
            #     cv2.putText(img, f'{NamesOfObjects[classId - 1].upper()} [** Normal Objects ** ]->  {round(conf * 100, 2)}',
            #                 (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 1, (0, 255, 0), 2)
            cv2.putText(img, f'{NamesOfObjects[classId - 1].upper()} {round(conf * 100, 2)}',
            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1, (0, 255, 0), 2)
    except:
        pass

    # to prevent auto off drone
    drone.send_rc_control(0,0,0,0)
    # video.write(img)
    img = cv2.resize(img, (500, 500))
    cv2.imshow("img",img)
    cv2.waitKey(1)

