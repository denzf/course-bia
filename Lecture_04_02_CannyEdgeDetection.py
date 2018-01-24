import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# %%

if __name__ == '__main__':
    
    print("Lecture 04")
    img_file_name = "GFP_06-DAPI.tif"
    print("... reading image from file: " + img_file_name)
    
    '''Convert image to array and use  matplotlib '''
    itkImage = sitk.ReadImage(img_file_name)
    #numpyArray = sitk.GetArrayFromImage(itkImage)
    #plt.imshow(numpyArray); plt.show()

    '''use helper function '''
    #ut.sitk_show(itkImage)

    itkImage = sitk.Cast(itkImage, sitk.sitkFloat32)
    #smooth = sitk.SmoothingRecursiveGaussian(itkImage ,4.0)
    #use helper function
    #ut.sitk_show(smooth)

    # Apply the Canny Edge Detector
    # Paramter 1: the input image
    # Parameter 2: the lower threshold (default set to 0.0)
    # Parameter 3: the upper threshold (default set to 0.0)
    # Parameter 4: the smoothing parameter as a vector (default set to (3, 0.0)    
    img_ce = sitk.CannyEdgeDetection(itkImage, 0.0, 2.0, (5, .75)) 
 
    '''Convert to Numpy Array '''
    numpyArray = sitk.GetArrayFromImage(img_ce)
    plt.imshow(numpyArray)
    plt.show()

    # Binary Threshold: http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1BinaryThresholdImageFilter.html
    # Paramter 1: the input image
    # Parameter 2: the lower threshold (default set to 0.0)
    # Parameter 3: the upper threshold (default set to 255.0)
    # Parameter 5: inside value    
    # Parameter 6: outside value
    # Note: Input Image Type is float and output is unsigned char (8 bits)
    img_Binary = sitk.BinaryThreshold(img_ce, 1, 1, 255, 0)

 
    fNameOut =  'CannyOutput.png'
    sitk.WriteImage(img_Binary,fNameOut)   

          
    #ut.sitk_show(img_ce)