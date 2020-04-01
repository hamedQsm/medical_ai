import pydicom as dicom
import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

outfile = "dataset_npy\\"
PathDicom = "dataset_dcm\\"

onlydirs = [d for d in listdir(PathDicom) if isdir(join(PathDicom, d))]

for d in onlydirs:
    lstFilesDCM = []  # create an empty list
    PathDicom += d
    outfile += d + '.npy'
    
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    RefDs = dicom.read_file(lstFilesDCM[0])    
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        if not filenameDCM.endswith('.dcm'):
            continue 
        ds = dicom.read_file(filenameDCM)    
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array 
    
    print(outfile)
    np.save(outfile, ArrayDicom)
    
    outfile = "dataset_npy\\"
    PathDicom = "dataset_dcm\\"

####################### test ###############################
from matplotlib import pyplot, cm

outfile += d + '.npy'
ArrayDicom = np.load(outfile)

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 200]))

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(z, x, np.flipud(ArrayDicom[:, 200, :]))

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(z, y, np.flipud(ArrayDicom[200, :, :]))

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

_x = []
_y = []
_z = []
threshold = np.quantile(np.array(ArrayDicom), .995)
for i, img in enumerate(ArrayDicom):
    indices = np.where(img>threshold)
    _x.extend(indices[0])
    _y.extend(indices[1])
    _z.extend(len(indices[0])*[i])

ax = plt.axes(projection='3d')
ax.scatter(_x, _y, _z, s=0.001, c='black')    