
import re
import os
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def renamefile(proj_height,output):
    #To rename
    cnt = 0
    yval = proj_height-1
    while(yval > -1):
        for xval in range(proj_height-1, -1,-1):
            filename = output + "/scene_" + str(xval).zfill(3) + "_" + str(yval).zfill(3) + ".jpg"
            os.rename(filename,output + "/scene_"+str(cnt)+".jpg")
            filename = output + "/scene_" + str(xval).zfill(3) + "_" + str(yval).zfill(3) + ".exr"
            os.rename(filename,output + "/scene_"+str(cnt)+".exr")
            cnt = cnt+1
        yval = yval-1
