#!/usr/bin/python

import numpy as np
import rospy
import os

from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

import argparse

from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

class HSVAreaSelectorNode:
    def __init__(self):

        rospy.loginfo("Creating HSVAreaSelectorNode")

        # init subscribers
        rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.rgb_image_cb)
        
        # initialize the list of reference points and boolean indicating
        # whether cropping is being performed or not
        self.refPt = []
        self.cropping = False
        
        self.counter = 0

        self.curr_rgb_img = None
        self.curr_hsv_img = None

        self.bridge = CvBridge()
        
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)    
        cv2.waitKey()  

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rospy.loginfo("SkinColorSelectorNode: Exiting")
        # close all open windows
        cv2.destroyAllWindows()

    def rgb_image_cb(self, rgb_img):
        if self.counter < 10:
            self.counter += 1
            return
            
        #rospy.loginfo("SkinColorSelectorNode: Processing RGB image")
        
        try:
            self.curr_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        # display the image and wait for a keypress
        cv2.imshow("image", self.curr_rgb_img)
        key = cv2.waitKey(10)
                
        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(self.refPt) == 2:
            roi = self.curr_rgb_img[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(roi_hsv)
            min_h, max_h, minloc, maxloc = cv2.minMaxLoc(h)
            plt.hist(h.ravel(),256,[0,256]); plt.show()
            mean_h, std_h = cv2.meanStdDev(h)
            print 'h_min=%f, h_max=%f, h_mean=%f, h_std=%f' % (min_h, max_h, mean_h[0], std_h[0])
            min_s, max_s, minloc, maxloc = cv2.minMaxLoc(s)
            mean_s, std_s = cv2.meanStdDev(s)
            print 's_min=%f, s_max=%f, s_mean=%f, s_std=%f' % (min_s, max_s, mean_s[0], std_s[0])
            min_v, max_v, minloc, maxloc = cv2.minMaxLoc(v)
            mean_v, std_v = cv2.meanStdDev(v)
            print 'v_min=%f, v_max=%f, v_mean=%f, v_std=%f' % (min_v, max_v, mean_v[0], std_v[0])
            cv2.imshow("skin", roi)
            cv2.waitKey(10)
            self.refPt = []
            self.cropping = False     
            
    def click_and_crop(self, event, x, y, flags, param):
       
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
          print("First point")
          self.refPt = [(x, y)]
          self.cropping = True
       
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
          # record the ending (x, y) coordinates and indicate that
          # the cropping operation is finished
          print("Second point")
          self.refPt.append((x, y))
          self.cropping = False
       
          # draw a rectangle around the region of interest
          cv2.rectangle(self.curr_rgb_img, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
          cv2.imshow("image", self.curr_rgb_img)

def main():
    rospy.init_node('hsv_area_selector')

    with HSVAreaSelectorNode() as ssn:
        rospy.spin()


if __name__ == "__main__":
    main()
