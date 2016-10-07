#!/usr/bin/python

import numpy as np
import rospy
import os

from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

class SkinDetectorNode:
    def __init__(self):

        rospy.loginfo("Creating SkinDetectorNode")

        # init subscribers
        rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.rgb_image_cb)

        self.curr_rgb_img = None
        self.curr_hsv_img = None

        self.bridge = CvBridge()

        self.skin_msk_pub = rospy.Publisher('/camera/rgb/image_rect_colorsub', Image, queue_size=10)
        
        self.skin_msk_img = Image()
        
        self.skin_lower = np.array([130, 10, 10], dtype = "uint8")
        self.skin_upper = np.array([180, 255, 255], dtype = "uint8")
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rospy.loginfo("SkinDetectorNode: Exiting")

    def rgb_image_cb(self, rgb_img):
        #rospy.loginfo("SkinDetectorNode: Received RGB image")
        try:
            self.curr_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        self.curr_hsv_img = cv2.cvtColor(self.curr_rgb_img, cv2.COLOR_BGR2HSV)
        skin_msk = cv2.inRange(self.curr_hsv_img, self.skin_lower, self.skin_upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        skin_msk = cv2.erode(skin_msk, kernel, iterations = 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        #Connect regions
        skin_msk = cv2.dilate(skin_msk, kernel, iterations = 2)
        
        #Remove regions
        skin_msk = cv2.erode(skin_msk, kernel, iterations = 3)
        
        #Enlarge regions
        skin_msk = cv2.dilate(skin_msk, kernel, iterations = 9)
        
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skin_msk = cv2.GaussianBlur(skin_msk, (3, 3), 0)
        
        #### Create Image ####
        self.skin_msk_img = self.bridge.cv2_to_imgmsg(skin_msk, "mono8")
       
        self.skin_msk_pub.publish(self.skin_msk_img)

def main():
    rospy.init_node('skin_detector_node')

    with SkinDetectorNode() as sdn:
        rospy.spin()


if __name__ == "__main__":
    main()
