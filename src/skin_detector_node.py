#!/usr/bin/python

import numpy as np
import argparse
import scipy
import rospy
from scipy.linalg import *
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import std_msgs.msg
from roslib import message
import sensor_msgs.point_cloud2
from std_msgs.msg import String
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import geometry_msgs.msg
import math
import PyKDL
import time
from time import sleep
import time
import math
import tf
import rospkg
import os

from sensor_msgs.msg import Image
import cv2
from std_msgs.msg import Float64
import tf
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

        self.skin_msk_pub = rospy.Publisher('/skin', Image, queue_size=10)
        
        self.skin_msk_img = Image()
        

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
        skin_msk = cv2.inRange(self.curr_hsv_img, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_msk = cv2.erode(skin_msk, kernel, iterations = 2)
        skin_msk = cv2.dilate(skin_msk, kernel, iterations = 2)
        
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skin_msk = cv2.GaussianBlur(skin_msk, (3, 3), 0)
        
        #### Create Image ####
        self.skin_msk_img = self.bridge.cv2_to_imgmsg(skin_msk, "8UC1")
       
        self.skin_msk_pub.publish(self.skin_msk_img)

def main():
    rospy.init_node('skin_detector_node')

    with SkinDetectorNode() as sdn:
        rospy.spin()


if __name__ == "__main__":
    main()
