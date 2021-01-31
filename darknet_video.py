from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

showCropedImageStatus = True
saveCropedImageStatus = True

def YOLO():
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./cfg/yolov3-tiny.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("/home/mumin/İndirilenler/VID_20191009_133209.mp4")
    #cap.set(3, 1280)
    #cap.set(4, 720)

    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #print(frame_size)
    #print(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))

    print("Starting the 'I4U' loop..")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(
                darknet.network_width(netMain),
                darknet.network_height(netMain),
                3)

    if cap.isOpened():
        print("Webcam online.")
    else:
        print("Webcam offline.")

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        # gelen frame bos ise while sonlaniyor
        if not ret:
        #if np.shape(frame_read) == ():
            break

        frame_read = cv2.flip(frame_read, 1)

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain), darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        imageToShowingCorped = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)


        #nparr = np.fromstring(img_str, np.uint8)
        #img1 = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for detection in detections:
            label = detection[0].decode('UTF-8') # b'person' sorunun cozmek icin artik person yaziyor
            confidence =  detection[1]
            bounds =  detection[2]

            if (showCropedImageStatus or saveCropedImageStatus):
                crop_img = darknet.CropPictureByBoundingBox(bounds, imageToShowingCorped)

            if (showCropedImageStatus == True):
                #cv2.imshow(str(label) + " oran: %" + str(confidence), crop_img)
                cv2.imshow(label, crop_img)

            if (saveCropedImageStatus == True):
                darknet.SaveCropedImage(label, crop_img)

        str = "FPS : %0.1f" % float(1 / (time.time() - prev_time))

        cv2.putText(image, str, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Showing I4U Video', image)
        key = cv2.waitKey(1)
        if (key == 27):
            break

    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
