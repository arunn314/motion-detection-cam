
import io
import os
import random
import sys
import threading
import time
import datetime

import picamera
from picamera.array import PiRGBArray

import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

from gdrive_handler import GDriveHandler
from utils import send_alert

notify_thresh = 10*60
delta_thresh = 25
min_area =  3000
blur_size = [21, 21]

frame_width = 320
frame_height = 240
path = '/home/pi/projects/pi-bot/recordings/'

def get_time(offset=0, with_second=True):
    """Get time formatted string for filenames."""
    today = datetime.datetime.now() - datetime.timedelta(seconds=offset)
    hour = str(today.hour)
    minute = str(today.minute)
    second = str(today.second)
    if with_second:
        return hour+'-'+minute+'-'+second
    else:
        return hour+'-'+minute

def get_max_area(contours):
    """Get the maximum area in all the contours."""
    max_area = 0
    for c in contours:
        temp = cv2.contourArea(c)
        if temp > max_area:
            max_area = temp

    return max_area

def motion_detect(frame, avg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, tuple(blur_size), 0)

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    cv2.accumulateWeighted(gray, avg, 0.01)

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, delta_thresh, 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    ret_contours = []
    for c in cnts:
        # if the contour is too small, ignore it
        # print cv2.contourArea(c)
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ret_contours.append(c)

    return (ret_contours, avg, gray, frame)


class Producer(threading.Thread):
    """Produces motion detected frames to a list."""
    def __init__(self, frames, condition):
        """
        Args:
            frames list of frames.
            condition synchronization object.
        """
        threading.Thread.__init__(self)
        self.frames = frames
        self.condition = condition
        self.last_notify = 0

        self.gray_refs = {'night': [], 'day': []}
        self.exclusions_path = '/home/pi/projects/pi-bot/exclusions/'
        self.get_exlusions()

    def get_exlusions(self):
        """Store exclusions images to avoid sending false positives.
        Exclusions such as sudden change in light conditions, bright lights.
        """
        files = os.listdir(self.exclusions_path)
        for filename in files:
            image = cv2.imread(self.exclusions_path + filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)

            if filename.startswith('day'):
                self.gray_refs['day'].append(gray)
            elif filename.startswith('night'):
                self.gray_refs['night'].append(gray)

    def day_or_night(self, gray_image, day_ref, night_ref):
        day_dist = ssim(gray_image, day_ref, multichannel=False)
        night_dist = ssim(gray_image, night_ref, multichannel=False)
        if day_dist < night_dist:
            return 'night'
        else:
            return 'day'

    def filter_images(self, images):
        """Filter false positives based Structural similarity index to
        avoid false alerts."""
        status = self.day_or_night(images[0][1],
                                   self.gray_refs['day'][0],
                                   self.gray_refs['night'][0])
        print status
        exclusions = self.gray_refs[status]
        threshold = 0.7
        last_ref = None
        result = []

        for filename, gray_img, raw_img in images:
            skip = False
            if last_ref:
                dist = ssim(gray_img, exclusions[last_ref], multichannel=False)
                if dist > threshold:
                    skip = True

            if not skip:
                for i, gray_ref in enumerate(exclusions):
                    if i == last_ref:
                        continue
                    dist = ssim(gray_img, gray_ref, multichannel=False)
                    if dist > threshold:
                        skip = True
                        last_ref = i
                        break

            if not skip:
                if (time.time() - self.last_notify) > notify_thresh:
                    send_alert('Alert! Motion detected near front door.')
                    self.last_notify = time.time()
                result.append((filename, gray_img, raw_img))
        return result


    def append_frames(self, motion_frames):
        """Append motion detected frames to the synchronized queue."""
        with self.condition:
            motion_frames = self.filter_images(motion_frames)
            for item in motion_frames:
                self.frames.append(item)
            sys.stdout.flush()
            self.condition.notify()

    def run(self):
        while True:
            try:
                self.capture()
            except Exception as e:
                print e

    def capture(self):
        """Capture and iterate Stream of images."""
        with picamera.PiCamera() as camera:
            # camera setup
            camera.resolution = (frame_width, frame_height)
            camera.framerate = 32
            camera.rotation = 90
            stream = PiRGBArray(camera, size=(frame_width, frame_height))

            # let camera warm up
            time.sleep(1)
            avg = None

            prev_area = 0
            upload_cnt = 0
            upload_threshold = 75
            motion_frames = []
            frame_cnt = 0

            start_time = time.time()

            print 'Ready'
            for frame in camera.capture_continuous(stream, 'bgr',
                                                   use_video_port=True):

                stream.seek(0)
                image = frame.array

                if avg is None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, tuple(blur_size), 0)
                    avg = gray.copy().astype("float")
                    stream.truncate()
                    continue

                (contours, avg, gray, image) = motion_detect(image, avg)

                #print contours
                if isinstance(contours, tuple):
                    contours = list(list(contours))
                if len(contours) > 0 and (time.time() - start_time) > 20:
                    if upload_cnt < upload_threshold:
                        print len(contours)
                        print str(datetime.datetime.now())
                        sys.stdout.flush()
                        time_str = get_time()
                        output_filename = path + 'img_' + time_str + '-' + str(frame_cnt) + '.jpg'
                        if frame_cnt % 3 == 0:
                            motion_frames.append((output_filename, gray, image))
                            upload_cnt += 1
                        frame_cnt += 1
                else:
                    upload_cnt = 0
                    if motion_frames:
                        if len(motion_frames) > 1:
                            self.append_frames(motion_frames)
                        motion_frames = []
                        frame_cnt = 0

                stream.seek(0)
                stream.truncate()


class Consumer(threading.Thread):
    """Consumes frames from a list and upload to Google Drive."""
    def __init__(self, frames, condition):
        """
        Args:
            frames list of frames.
            condition synchronization object.
        """
        threading.Thread.__init__(self)
        self.frames = frames
        self.condition = condition
        self.gdrive = GDriveHandler()

    def run(self):
        while True:
            try:
                with self.condition:
                    while True:
                        if self.frames:
                            output_filename, gray_image, image = self.frames.pop()
                            cv2.imwrite(output_filename, image)
                            self.gdrive.upload_file(output_filename)
                            os.remove(output_filename)
                            break
                        self.condition.wait()
            except Exception as e:
                print e
                self.gdrive = GDriveHandler()

def main():
    """Main Function"""
    frames = []
    condition = threading.Condition()
    t1 = Producer(frames, condition)
    t2 = Consumer(frames, condition)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':
    global min_area

    cnt = len(sys.argv)
    if cnt > 1:
        min_area = int(sys.argv[1])

    main()
