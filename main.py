import xmlrpc.client
# import time
from geopy.geocoders import Nominatim
# import geocoder
import requests

def get_precise_location(api_key):
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + api_key
    data = {"considerIp": "true"}  # Use IP address to approximate location if no other information is available
    response = requests.post(url, json=data)
    location_data = response.json()
    return location_data

api_key = 'AIzaSyCua11jotMEatjUiQLjlLdsV7T4lHSCHI4'

url = 'http://erp1.yushengstudio.net/'
db = 'mydbs'
username = 'admin'
password = 'admin'

# get location
geolocator = Nominatim(user_agent="xmlRPC")

common = xmlrpc.client.ServerProxy('{}/xmlrpc/2/common'.format(url))
print(common.version())
# login to server
uid = common.authenticate(db, username, password, {})

if uid == 2:
    print("uid:", 2, "login success.") # Check access expected to print 2

models = xmlrpc.client.ServerProxy('{}/xmlrpc/2/object'.format(url))

import argparse
import datetime
import math
import sys
import os

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage, Log,
                          cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--snapshots", type=str, default="images/test/detections", help="output directory of detection snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# make sure the snapshots dir exists
os.makedirs(args.snapshots, exist_ok=True)

# create video output object 
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network

# detect plate
netp = detectNet(model="networks/az_plate/az_plate_ssdmobilenetv1.onnx", labels="networks/az_plate/labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes", threshold=0.5)
# detect number
netn = detectNet(model="networks/az_ocr/az_ocr_ssdmobilenetv1_3.onnx", labels="networks/az_ocr/labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes", threshold=0.5)

# create video sources
input = videoSource(args.input, argv=sys.argv)

def itoa(n):
    if n < 11:
        return str(n - 1)
    return chr(n+54)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # detect objects in the image (with overlay)
    detections = netp.Detect(img, overlay="lines")
    str1 = ''
    # map_count = {}
    for idx, detection in enumerate(detections):
        # print(detection)
        roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
        snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
        cudaCrop(img, snapshot, roi)
        cudaDeviceSynchronize()
        # saveImage(os.path.join(args.snapshots, f"{timestamp}-{idx}.jpg"), snapshot)
        detect_number = netn.Detect(snapshot, overlay="lines")
        maps = []
        for obj in detect_number:
            maps.append((obj.Left, itoa(obj.ClassID)))
        maps.sort(key=lambda x: x[0])
        temp = [ele[1] for ele in maps]
        if temp and len(temp) == 7:
            str1 = ''.join(temp)
            # print(str1)
        # for numb in detect_number
        #print("detected {:d} number in image".format(len(detect_number)))
        del snapshot
    if str1 != '':
        read_plate = str1
        ids = models.execute_kw(db, uid, password, 'vehicle.property', 'search', [[['license_plate', '=', read_plate]]])
        if not ids:
            print("No found")
            continue
        record = models.execute_kw(db, uid, password, 'vehicle.property', 'read', [ids])
        if record: # if record exist
            # create location record
            location = get_precise_location(api_key)
            latitude = location['location']['lat']
            longitude = location['location']['lng']
            location = geolocator.reverse(f"{latitude}, {longitude}")
            call = models.execute_kw(db, uid, password, 'vehicle.location', 'create',
                                   [{'latitude': latitude, 'longitude': longitude, 'name': location.address, 'vehicle_id': ids[0]}])
            print("Vehicle discovered", record)
    # if str1 in map_count:
        # map_count[str1] = map_count[str1] + 1
    # else:
        # map_count[str1] = 1
    # print(max(map_count.keys()))
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, netp.GetNetworkFPS()))

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
        
