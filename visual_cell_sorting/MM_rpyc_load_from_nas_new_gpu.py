import sys, os
import time
import copy
import numpy as np
import rpyc
import socket
import base64, zlib
import clr, System
import ctypes
import skimage
from System import Array, Byte
from System.Runtime.InteropServices import GCHandle, GCHandleType

def Startup(param):
    Docommand(param)

def Docommand(param):
    start0 = time.time()
    writeTimerMessages = False
    mask_save_dir = "Y:\\2023\\Hyeon-Jin\\10-02-23_F1-E11-12-SC35-plate5-B04_masks"
    imageName = "Camera GFP"
    
    #imageHandle = 0
    #imageHeight = 0
    #imageWidth = 0
    #imageDepth = 0
    
    
    # Gather information about the original image
    #_,imageHandle = mm.GetCurrentImage(imageHandle)

    #_,_,imageName = mm.GetImageName(imageHandle, imageName)
    #_,_,imageHeight = mm.GetHeight(imageHandle, imageHeight)
    #_,_,imageWidth = mm.GetWidth(imageHandle, imageWidth)

    #pixels = mm.GetImagePixels(imageHandle)

    # Get height and width
    width = 1739 #pixels.GetLength(0) #1739
    height = 1536 #pixels.GetLength(1) #1536
    
    #if writeTimerMessages:
    #     mm.PrintMsg("Convert from numpy array timer: " + str((time.time()-start0)*1000))
    
    start = time.time()
    while True:
        try:
            conn = rpyc.connect("128.208.26.4", 18871)
            break
        except socket.timeout:
            mm.PrintMsg("Caught a timeout! Reconnecting...")
            time.sleep(2)
    conn._config['sync_request_timeout'] = 10000 #set timeout to 10k seconds
    bgsrv = rpyc.BgServingThread(conn)
    mon = conn.root.ImageAnalysis(get_outputmask)
    b64_output_mask = mon.exposed_run_pipeline_on_image()
    np_output_mask = skimage.io.imread(os.path.join(mask_save_dir, "mask" + str(b64_output_mask) + ".png"))
    bgsrv.stop()
    conn.close()
    end = time.time()

    if writeTimerMessages:
         mm.PrintMsg("Send and receive time: " + str((end-start)*1000))

    # Convert back from numpy array
    start = time.time()
    if not np_output_mask.flags.f_contiguous:
        np_output_mask = np_output_mask.copy(order='F')
    output_mask = Array.CreateInstance(System.Byte, width, height)
    destHandle = GCHandle.Alloc(output_mask, GCHandleType.Pinned)
    sourcePtr = np_output_mask.__array_interface__['data'][0]
    destPtr = destHandle.AddrOfPinnedObject().ToInt64()
    ctypes.memmove(destPtr, sourcePtr, np_output_mask.nbytes)
    end=time.time()

    #if writeTimerMessages:
    #     mm.PrintMsg("Convert from numpy array timer: " + str((end-start)*1000))
         
    
    # Generate a new image
    newImageHandle = 0
    newImageName = imageName + "_Binary"
    _,_,_,_,_,newImageHandle = mm.CreateImage(width, height, 8, newImageName, newImageHandle)

    # Write image
    start = time.time()
    _ = mm.WriteImage(newImageHandle, 0, 0, width, height, 8, 0, 0, output_mask)
    end = time.time()
    if destHandle.IsAllocated: destHandle.Free()
    end0 = time.time()

    if writeTimerMessages:
        mm.PrintMsg("Save Image Timer: " + str((end-start)*1000))
        mm.PrintMsg("Total Time: " + str((end0-start0)*1000))
        mm.PrintMsg("--------------------------------------------------------------------")

def Shutdown():
    pass

def get_outputmask(image_name, output_mask):
    #....
    print("call back in client", image_name)
