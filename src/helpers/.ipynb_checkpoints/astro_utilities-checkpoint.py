#--------------------------------------Imports
from photutils import detect_threshold
#2D Circular Gaussian Kernel to smooth the data prior to Thresholding
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.nddata.utils import PartialOverlapError
#Finetuning objects
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
norm = ImageNormalize(stretch=SqrtStretch())

from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties

#Imports
import astropy.units as u
from photutils import EllipticalAperture
from astropy.nddata.utils import Cutout2D
from astropy.io import fits


from random import sample
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle


#-------------------------------------------------PATHS
current_path = os.getcwd()
cadc_im_dir = os.path.join(current_path, 'CADC_Images/')  #To store downloaded .fz files

cadc_down_dir = os.path.join(current_path, 'data/CADC_downloads/')  #To store downloaded .fz files via SYS 
                                                               #                    (for CFHT files)


dict_savepath = os.path.join(current_path,'results/tmp/vector_bins/')  #Toy Data DICTs





#-----------------------------------Saving and loading objects-----------------------------
def save_obj(obj, name):
    with open(dict_savepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(dict_savepath  + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
   
    
    
    
    
    


#----------------------------------------------FITS File Manipulation----------------------------------

'''
Reads in FZ ext files from directory
and returns list of filenames
'''
def load_fits_filenames():
    #Set correct path
    fits_filenames = glob(cadc_im_dir + '/' + '*.fz')
    #Ensure data was read in correctly
    for idx, name in enumerate(fits_filenames):
        #Store image data
        image_data = fits.getdata(name, ext= 0)
        print('ID:'+str(idx)+ ' \t Name of file: '+name+ '\n Shape: {}'.format(image_data.shape))
    
    #Return list
    return fits_filenames


'''
This function correctly sets up file path to load given obs_ID

Given obs_ID will be converted to STRING!

Loads the HDU List

'''
def load_hdu_list_for_ID(obs_ID):
    
    #HARDCODE Path for downloading
    cadc_down_dir =  "C:\\Users\\Ahnaf Tazwar Ayub\\Documents\\NRC-Work\\project\\cadc-im\\data\\CADC_downloads"
    
    #Cast explicitly as String
    obs_ID = str(obs_ID)
    #The filename with the appropriate extension
    file_to_open = obs_ID + 'p.fits.fz' #Extension for CFHT downloads  
    #Load from
    filepath = os.path.join(cadc_down_dir , file_to_open)

    #Extract HDU List
    hdu_list = fits.open(filepath)   
    return hdu_list




'''
Download File from CADC
given an OBS_ID
'''
def download_fits_file(obs_ID):   
    #Cast as String
    obs_ID = str(obs_ID)
    #Setup download URL
    url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/"+obs_ID+"p.fits.fz"
    print(url)
    #--Run download
    #os.system(('cd data/CADC_downloads/ && '+ 'curl -O -J -L '+url))
    os.system(('cd .. && cd data/CADC_downloads/ && '+ 'curl -O -J -L '+url))

    print("Download successful")
    return




#---------------------------- New utility functions -------------------
    
'''
DOWNLOAD a single observation

-- SPECIFY Class(from IDX interval) as a STRING

'''   
def download_single_obs(obs_ID, class_dir):  
    #Cast as string
    obs_ID = str(obs_ID)   
    #URL
    url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/"+ obs_ID + "p.fits.fz"

    #Specify target folder(eg. c1)
    root =  "C:\\Users\\Ahnaf Tazwar Ayub\\Documents\\NRC-Work\\project\\cadc-im\\"
    
    target_dir = os.path.join(root, "data\\validation", class_dir)
    print("Target : {}".format(target_dir))

    #Navigation command
    navigate = 'cd ' + target_dir
    print("Navigation command: {}".format(navigate))
    #Download command
    download = 'curl -O -J -L '+ url
    #Navigate and delete
    os.system(( navigate + ' && ' + download))
    print("Finished downloading")
    return


'''
REMOVE a single observation
'''   
def remove_single_obs(obs_ID, class_dir):
    #Cast as string
    obs_ID = str(obs_ID)   
    #Filename
    fname = obs_ID + "p.fits.fz"

    #Specify target folder(eg. c1)
    root =  "C:\\Users\\Ahnaf Tazwar Ayub\\Documents\\NRC-Work\\project\\cadc-im\\"
    
    target_dir = os.path.join(root, "data\\validation", class_dir)
    print("Target : {}".format(target_dir))

    #Navigation command
    navigate = 'cd ' + target_dir
    print("Navigation command: {}".format(navigate))
    #Download command
    delete = 'del '+ fname
    #Navigate and delete
    os.system(( navigate + ' && ' + delete))
    print("Finished deleting file")
    return



'''
For VALIDATION data

This function correctly sets up file path to load given obs_ID
Given obs_ID will be converted to STRING!
Loads the HDU List

'''
def load_hdu_list_for_ID_validation(obs_ID, class_dir):
    
    #HARDCODE Path for downloading
    cadc_down_dir =  "C:\\Users\\Ahnaf Tazwar Ayub\\Documents\\NRC-Work\\project\\cadc-im\\data\\validation\\"
    cadc_down_dir = os.path.join(cadc_down_dir, class_dir)
    
    #Cast explicitly as String
    obs_ID = str(obs_ID)
    #The filename with the appropriate extension
    file_to_open = obs_ID + 'p.fits.fz' #Extension for CFHT downloads  
    #Load from
    filepath = os.path.join(cadc_down_dir , file_to_open)

    #Extract HDU List
    hdu_list = fits.open(filepath)   
    return hdu_list




'''
Pass in Fits Filenames LIST and IDX of file# to extract from

CLASSIC: using an entire directory

'''
def load_hdu_list(fits_filenames, idx):
    #Extract HDU List
    hdu_list = fits.open(fits_filenames[idx])   
    return hdu_list






'''
Pass in a HDU List and an idx(to specify CCD to extract from)

Returns Loaded Data.

CCD - actually from (0-->36)
 so passing inidx 36 means we are extracting ccd35(which is CCD 36 since the 
 numbers start from 0 in HDU List)

'''

#Load 1 CCD
def extract_CCD_data(hdu_list, idx):
    #Obtain data
    X = hdu_list[idx].data
    print("CCD-{} loaded. Shape: {}".format(idx, X.shape))

    return X

#---------------------------------------------------------Source Detection -----------------------------------
'''
Pass in Image data as MATRIX
--Specify nsigma and Connected Points

--Find Sources WITHOUT Deblending

-- returns a SegmentationImage object

'''
def get_non_deblended_sources(data, nsigma, connected_points):
    #Set Detection Threshold
    threshold = detect_threshold(data, nsigma = nsigma)
    #Use a FWHM of 3 pixels
    sigma = 3.0 * gaussian_fwhm_to_sigma
    #Setup Filter Kernel
    kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
    kernel.normalize()
    #Carry out Segmentation
    segments = detect_sources(data, threshold,
                              npixels = connected_points, 
                              filter_kernel = kernel)

    return segments



'''
Pass in Image data as MATRIX

--Specify nsigma and Connected Points
--Find Sources WITH Deblending

-- returns a SegmentationImage object

'''
def get_deblended_sources(data, nsigma, connected_points):
    #Set Detection Threshold
    threshold = detect_threshold(data, nsigma = nsigma)
    #Use a FWHM of 3 pixels
    sigma = 3.0 * gaussian_fwhm_to_sigma
    #Setup Filter Kernel
    kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
    kernel.normalize()
    #Carry out Segmentation
    segments = detect_sources(data, threshold, npixels = connected_points, filter_kernel = kernel)

    #DEBLENDED Sources
    # Must use same npixels and Filter Kernel as before
    segm_deblended = deblend_sources(data, segments,
                        filter_kernel=kernel,
                        npixels = connected_points,
                        nlevels = 32, #Number of multi-thresholding levels to use
                        contrast = 0.9) #[0,1] where 0 is MAXIMUM DEBLENDING and 1 is none

    return segm_deblended






'''
Pass in Image data as MATRIX, along with SegmenationImage(either Deblended or ND)

- Plots Detected Sources

'''
def plot_detected_sources(data, deblended_sources):
    
    segm_deblended = deblended_sources
    
    #Find the Table of Sources
    cat = source_properties(data, segm_deblended)
    table = cat.to_table()
    #Format data
    table['xcentroid'].info.format ='%.8g'
    table['ycentroid'].info.format ='%.8g'
    #Define an approximate extent for each isophotal entity
    r = 3
    
    #Iterate over each object in cat (NOT THE TABLE!)
    #Define empty container
    apertures =[]
    
    for obj in cat:
        #Define Ellipse properties
        position = np.transpose((obj.xcentroid.value , obj.ycentroid.value))
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        #Angle (Convert to Radians)
        theta = obj.orientation.to(u.rad).value
        #Form and ADD
        aperture = EllipticalAperture(position, a , b, theta = theta)
        apertures.append(aperture)
        
    fig, (ax1, ax2) = plt.subplots(1, 2 , figsize =(20,10))
    #Plot Original Image
    ax1.imshow(data, cmap='gray' , norm = norm) #norm as Before
    ax1.set_title('Original Image')
    #Plot Segmented Image
    #Make color map
    cmap = segm_deblended.make_cmap(random_state= 123 )
    ax2.imshow(segm_deblended, cmap=cmap)
    ax2.set_title('Segmented Image')
    
    #Plot Apertures
    for aperture in apertures:
        aperture.plot(axes = ax1, lw =1.5, color ='white')
        aperture.plot(axes = ax2, lw =1.5, color ='white')
        
    #---------PLOT
    filename = 'detected_sources.png'
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/astro/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("Detected Sources saved as {} at {}".format(filename, im_savepath))

    

    
'''
Simply logs time
'''
def time_logger(start, end):
    time_taken = np.around((end-start),2)
    print("Time taken: {} seconds.".format(time_taken))
    print("\n")

    
'''
Pass in Image data as MATRIX, 
along with Deblended & Non-Deblended SegmentationObject

- Plots Original / Deblended/ NonDeblended  SIDE by SIDE

'''    
def plot_segmentations(data, 
                       segments, 
                       segm_deblended):
    
    #--------------PLOT-----------
    fig, (ax1 , ax2 , ax3) = plt.subplots(1,3, figsize =(20,10))
    #Plot Original Image
    ax1.imshow(data, cmap='gray' , norm = norm)
    ax1.set_title('Original Image')
    #Plot Segmented Image
    #Make color map
    cmap = segments.make_cmap(random_state= 123 )
    ax2.imshow(segments, cmap=cmap)
    ax2.set_title('Segmented Image')
    #Plot Deblemded Segments
    cmap = segm_deblended.make_cmap(random_state= 12345 )
    ax3.imshow(segm_deblended, cmap=cmap)
    ax3.set_title('Segmented Image with Deblending')
    
    #---------PLOT
    filename = 'side_by_side.png'
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/astro/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("SIDE by SIDE saved as {} at {}".format(filename, im_savepath))

    
    
'''
Params---

Params---
1. Full IMG - X matrix
2. nsigma
3. connected_points = npixels
RETURNS a Deblended Source Object
'''
def get_deblended_sources_timed(data, nsigma, connected_points):
    #Set Detection Threshold
    threshold = detect_threshold(data, nsigma = nsigma)
    #Use a FWHM of 3 pixels
    sigma = 3.0 * gaussian_fwhm_to_sigma
    #Setup Filter Kernel
    kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
    kernel.normalize()
    
    #Time
    start = timer()
    
    #Carry out Segmentation
    segments = detect_sources(data, threshold, npixels = connected_points, filter_kernel = kernel)

    #DEBLENDED Sources
    # Must use same npixels and Filter Kernel as before
    segm_deblended = deblend_sources(data,    #X passed in
                                     segments, #non-deblended
                                     filter_kernel=kernel,
                                    npixels = connected_points,
                                    nlevels = 32, #Number of multi-thresholding levels to use
                                    contrast = 0.001) #[0,1] where 0 is MAXIMUM DEBLENDING
    end = timer()#End
    time_logger(start, end) #Log

    return segm_deblended


'''
Take in Deblended Sources & Original Data
--Find Table and form cutouts
-- Impute NON-Overlapping portions with NAN

Returns a List of Cutout2D Objects [32x32 each]
'''
def get_cutouts_list(data , deblended_sources):
    #Find the Table of Sources
    cat = source_properties(data, deblended_sources)
    table = cat.to_table()
    
    #Format data
    table['xcentroid'].info.format ='%.8g'
    table['ycentroid'].info.format ='%.8g'
    #Retrieve Centroids
    centroids = np.transpose((table['xcentroid'],table['ycentroid']))
    x_centroids = centroids[:, 0]
    y_centroids = centroids[:, 1]
    #Form Cutouts
    cutouts = []
    
    #Make cutouts
    for i in range(len(centroids)):
        
        #Coordinates
        coords = (x_centroids[i], y_centroids[i])
        
        try:
            #Form 32x32 cutouts
            size = (32, 32)
            cutout = Cutout2D(data,
                              coords,
                              size,
            #Fill value choices: nan/0                  
            fill_value = np.nan,                          
            mode = 'strict'
            ) #CHOOSE FILL_VALUE to be nan
            #List of Cutouts
            cutouts.append(cutout)
        
        except PartialOverlapError:
            #Couldn't form cutout
            print("Source is on/near edge! \n")
                            
    print("# Sources Detected: {} ".format(len(cutouts)))
    return cutouts

'''
Function which checks if given coordinates lies within our Image DATA (2D matrix)

RETURNS TRUE if it does
'''
def within_image(coords, data):
    #Extract x,y
    x = coords[0]
    y = coords[1]
    
    #Define boundaries
    w = data.shape[0]
    h = data.shape[1] #NUMPY Convention! x is downwards

    
    #Define tolerance width(x/2)
    tolerance = 16  
    
    #X-limits
    x1 = tolerance
    x2 = h - tolerance
    
    #Y-limits
    y1 = tolerance  
    y2 = w - tolerance
     
    if(x < x1):
        print("Left edge Limit: {}".format(x1))
        print("x: {}, y: {}".format(x,y))
        print("Outside edge \n")
        return False
    elif(x > x2):
        print("Right edge Limit: {}".format(x2))
        print("x: {}, y: {}".format(x,y))
        print("Outside edge \n")
        return False
    elif(y < y1):
        print("UPPER edge Limit: {}".format(y1))
        print("x: {}, y: {}".format(x,y))
        print("Outside edge \n")
        return False
    elif(y > y2):
        print("LOWER edge Limit: {}".format(y2))
        print("x: {}, y: {}".format(x,y))
        print("Outside edge \n")
        return False
    else:
        #print("Within image \n")
        return True




'''
Pass in a list of Cutouts and obtain a 2D matrix 

(Shape: num_cutouts x 1024)
'''
def create_matrix(cutouts):
    nests = []
    for c in cutouts:
        #Flatten data
        c = c.data.flatten()
        nests.append(c)
        #Convert to Array
    ret_matrix = np.array(nests)
    return ret_matrix




'''
Pass in a CUTOUT DATA MATRIX and impute nans with MEDIAN
RETURNS Matrix Data, with imputed values
'''
def impute_with_median(X):
    #Find median
    median = np.nanmedian(X)
    #Locate nan and impute
    X[np.where(np.isnan(X))] = median
    return X



    
'''
Pass in dimesnions of Grid: eg. 2,3,4,5

Pass in a list of cutouts.
'''
def plot_cutouts_grid(grid_dim, cutouts):
    
    fig = plt.figure(figsize = (15,10)) # Setup

    #randomly sample
    sampled = sample(cutouts, np.power(grid_dim , 2))

    for idx, cutout in enumerate(sampled):
        #Get data
        data = cutout.data
        #Setup dimensions
        dim = int(np.sqrt(len(sampled)))
        #Add axis
        axis = fig.add_subplot(grid_dim, grid_dim, idx + 1)
        #axis.hist(data, bins = 10)
        axis.imshow(data, cmap ='gray')
    #Layout
    #plt.title("{}x{} Grid of Histograms".format(grid_dim, grid_dim))

    fig.tight_layout()
    
    #---------PLOT
    filename = "{}x{}_cutouts".format(grid_dim, grid_dim)
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/astro/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("Cutouts Grid saved as {} at {}".format(filename, im_savepath))   
    
   


'''
Pass in Image data as MATRIX, along with SegmenationImage(either Deblended or ND)

- Requires list of edge_coords

- Plots Detected Sources
-Plots edge sources as well

'''
def plot_edge_sources(data, deblended_sources, ccd_num):
    
    segm_deblended = deblended_sources
    
    #Find the Table of Sources
    cat = source_properties(data, segm_deblended)
    table = cat.to_table()
    #Format data
    table['xcentroid'].info.format ='%.8g'
    table['ycentroid'].info.format ='%.8g'
    #Define an approximate extent for each isophotal entity
    r = 3
    
    #Iterate over each object in cat (NOT THE TABLE!)
    #Define empty container
    apertures =[]
   
    edge_coords = []
    edge_apertures = []
    
    for obj in cat:
        #Define Ellipse properties
        position = np.transpose((obj.xcentroid.value , obj.ycentroid.value))
        
        #Check if edge
        if(not(within_image((obj.xcentroid.value , obj.ycentroid.value),
                            data))):
            #It is edge
            coords = (obj.xcentroid.value , obj.ycentroid.value)
            edge_coords.append(coords)
            
            continue
            
       #Otherwise Form aperture
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        #Angle (Convert to Radians)
        theta = obj.orientation.to(u.rad).value
        #Form and ADD
        aperture = EllipticalAperture(position, a , b, theta = theta)
        apertures.append(aperture)
       
    #Form edge apertures
    for edge_coord in edge_coords:
        #Define Ellipse properties
        position = edge_coord
        a = 16
        b = 16
        #Angle (Convert to Radians)
        #theta = obj.orientation.to(u.rad).value
        #Form and ADD
        aperture = EllipticalAperture(position, a , b)
        edge_apertures.append(aperture)
   
        
        
    fig, (ax1, ax2) = plt.subplots(1, 2 , figsize =(20,20))
    #Plot Original Image
    ax1.imshow(data, cmap='gray' , norm = norm) #norm as Before
    ax1.set_title('Original Image')
    #Plot Segmented Image
    #Make color map
    cmap = segm_deblended.make_cmap(random_state= 123 )
    ax2.imshow(segm_deblended, cmap=cmap)
    ax2.set_title('Segmented Image')
    
    #Plot Apertures
    for aperture in apertures:
        aperture.plot(axes = ax1, lw =1.5, color ='white')
        aperture.plot(axes = ax2, lw =1.5, color ='white')
        
    #Plot Apertures
    for aperture in edge_apertures:
        aperture.plot(axes = ax1, lw =1.5, color ='red')
        aperture.plot(axes = ax2, lw =1.5, color ='red')
        
    #---------PLOT
    filename = 'edge_sources_'+ str(ccd_num) +'.png'
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/astro/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("Edge Sources saved as {} at {}".format(filename, im_savepath))
    
 



