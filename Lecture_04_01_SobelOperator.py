import SimpleITK as sitk
print(sitk.Version())

import numpy as np
import matplotlib.pyplot as plt
#import utils as ut

# %%

if __name__ == '__main__':

    #outPath = r'Lecture_04/'
    
    print("Lecture 04")
    
    img_file_name = "GFP_06-DAPI.tif"
    print("... reading image from file: " + img_file_name)
    
    '''Convert image to array and use matplotlib '''
    itkImage = sitk.ReadImage(img_file_name)
    numpyArray = sitk.GetArrayFromImage(itkImage)
    plt.imshow(numpyArray)
    plt.show()

    itkImage = sitk.Cast(itkImage, sitk.sitkFloat32)
       
    smooth = sitk.SmoothingRecursiveGaussian(itkImage,4.0)
    fNameOut =  'SmoothingRecursiveGaussian.tif'
    sitk.WriteImage(sitk.Cast(smooth, sitk.sitkUInt8), fNameOut)   
    plt.imshow(  sitk.GetArrayFromImage(smooth)  )
    plt.show()

    sobelImage = sitk.SobelEdgeDetection(itkImage)
    fNameOut =  'SobelEdgeDetection.tif'
    sitk.WriteImage(sitk.Cast(sobelImage, sitk.sitkUInt8), fNameOut)   

    sobelImageSmooth = sitk.SobelEdgeDetection(smooth)
    fNameOut =  'SobelEdgeDetectionSmoothingRecursiveGaussian.tif'
    sitk.WriteImage(sitk.Cast(sobelImageSmooth, sitk.sitkUInt8), fNameOut)   
    plt.imshow(  sitk.GetArrayFromImage(sobelImageSmooth)  )
    plt.show()
