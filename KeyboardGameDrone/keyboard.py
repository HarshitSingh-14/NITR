from djitellopy import tello
from time import sleep
import keyCapture as key
import cv2
import time

key.init()
drone = tello.Tello()
drone.connect()
global image




def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 33
    if key.getKey("LEFT"):
        lr = -speed*3
    elif key.getKey("RIGHT"):
        lr = speed*3


    if key.getKey("UP"):
        fb = speed*3
    elif key.getKey("DOWN"):
        fb = -speed*3





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

drone.streamon()


while True:
    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (500, 500))
    cv2.imshow("Img", img)
    cv2.waitKey(1)