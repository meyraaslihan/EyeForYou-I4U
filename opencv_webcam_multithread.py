#!/usr/bin/env python

from threading import Thread, Lock
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.fps = 0

        self.showCropedImageStatus = True
        self.saveCropedImageStatus = False

        self.netMain = None
        self.metaMain = None
        self.altNames = None

        self.configPath = "./cfg/yolov3-tiny.cfg"
        self.weightPath = "./cfg/yolov3-tiny.weights"
        self.metaPath = "./cfg/coco.data"
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
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
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.stream = cv2.VideoCapture(src)
        #self.stream = cv2.VideoCapture("/home/mumin/İndirilenler/VID_20191009_133209.mp4")
        #self.stream.set(3, 1280)
        #self.stream.set(4, 720)

        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

        frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #print(frame_size)
        #print(cv2.CAP_PROP_FPS)

        #out = cv2.VideoWriter(
        #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        #    (darknet.network_width(netMain), darknet.network_height(netMain)))

        print("Starting the 'I4U' loop..")

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(
                    darknet.network_width(self.netMain),
                    darknet.network_height(self.netMain),
                    3)

        if self.stream.isOpened():
            print("Webcam online.")
        else:
            print("Webcam offline.")

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.detect_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame # alt taraf eskiden vardı
        #frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self, detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
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

    def StuffForCroped(self, detections, imageToShowingCorped):
        for detection in detections:
            label = detection[0].decode('UTF-8') # b'person' sorunun cozmek icin artik person yaziyor
            confidence = detection[1]
            bounds = detection[2]

            if (self.showCropedImageStatus or self.saveCropedImageStatus):
                crop_img = darknet.CropPictureByBoundingBox(bounds, imageToShowingCorped)

            if (self.showCropedImageStatus == True):
                showCropedThread = Thread(target=cv2.imshow(label, crop_img))
                showCropedThread.start()
                #cv2.imshow(str(label) + " oran: %" + str(confidence), crop_img)

            if (self.saveCropedImageStatus == True):
                saveCropedThread = Thread(target=darknet.SaveCropedImage(label, crop_img))
                saveCropedThread.start()

    def work(self):
        frame = self.read()

        fps_str = "FPS : %0.1f" % self.fps
        cv2.putText(frame, fps_str, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # gelen frame bos ise while sonlaniyor
        #if not grabbed:
        #if np.shape(frame) == ():
        #    break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(vs.netMain), darknet.network_height(vs.netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        imageToShowingCorped = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        darknet.copy_image_from_bytes(vs.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(vs.netMain, vs.metaMain, vs.darknet_image, thresh=0.25)

        #print(vs.darknet_image)

        #nparr = np.fromstring(img_str, np.uint8)
        #img1 = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        image = vs.cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tforstuff = Thread(target=self.StuffForCroped(detections, imageToShowingCorped))
        tforstuff.start()

        cv2.imshow('Showing I4U Video With Thread', image)

def work():
    while True :
        prev_time = time.time()
        vs.work()

        if cv2.waitKey(1) == 27 :
            break

        vs.fps = float(1 / (time.time() - prev_time))

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    #vs = WebcamVideoStream(src="/home/mumin/İndirilenler/VID_20191009_133209.mp4").start()

    thread = Thread(target=work())
    thread.start()
