#------------------- Module for analyzing elongatin/ellipticity/area/Rotational Invariance
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


    
#-------------------------------------------------Statistical Analysis-------------------------------------
'''
Takes in a SOURCE Image from 'Closest Samples'
 and the appropriate SegmentationImage(sources) for that source (Non-Deblended Ieally)
 
RETURNS a tuple of (Area, Ellipticity, Elongation)
or NONE if Can't detect sources
'''

def find_features(data, sources, warn =True):

    #sources is a SegmentationImage object
    if(sources==None):
        #Warn if need be
        if(warn):
            print("Couldn't detect source!")
        return None
    
    #Find the Table of Sources
    cat = source_properties(data, sources)
    table = cat.to_table()
    #Format data and retain useful ones
    table['ellipticity'].info.format ='%.8g'
    table['elongation'].info.format ='%.8g'
    table['area'].info.format ='%.8g'
    
    #Features to retain
    features = ['area','ellipticity','elongation' ]
    
    #Get data
    area_vals = cat.area.value
    ellip_vals = cat.ellipticity.value
    elongation_vals = cat.elongation.value

    #Print
#     print("Area: {}".format(area_vals))
#     print("Ellipticty: {}".format(ellip_vals))
#     print("Elongation_vals: {}".format(elongation_vals))
    
#     print(table[features])
    
    #Return Values
    area = np.mean(area_vals)
    ellipticity = np.mean(ellip_vals)
    elongation = np.mean(elongation_vals)
    
    return (area, ellipticity, elongation)

'''
Requires Distance MAP
and Grid NODE COORDS

Returns [(0,0),(0,0),(0,0),] if couldn't detect source

otw Returns List of tuples : Area, Ellipticity Elongation
'''

def summary_stats_for_bmu(sample_coord, distance_map ,
                         warn = True): 
    #For one BMU
    #sample_coord =(4,14)
    #Find Closest Samples
    closest_samples = find_closest_samples(sample_coord,
                                          distance_map)
    
    #Failed ID Tracker
    failed_idxs = []

    #Big collections for BMU Node
    area_vals = []
    ellipticity_vals =[]
    elongation_vals =[]

    #Start timer
    start = timer()
    
    #Loop over samples
    for idx, similar_sample in enumerate(closest_samples):
        #Run Source Segmentation
        non_deblended_segments = get_non_deblended_sources(similar_sample, #Pass in the sample
                                                           nsigma = 3, 
                                                           connected_points =  5)
        #Get tuple of features
        features = find_features(similar_sample, non_deblended_segments,
                                warn)

        #Check if we couldn't detect 
        if(features == None):
            #Impute with NaN
            area_vals.append(np.nan)
            ellipticity_vals.append(np.nan)

            elongation_vals.append(np.nan)
            
            #To impute later with mean of rest
            failed_idxs.append(idx)
        else:
             #Append to collections
            area_vals.append(features[0])
            ellipticity_vals.append(features[1])
            elongation_vals.append(features[2])
            
            
            
        #Imputation value: MEAN avoid NAN vals
        imputation_val_area = np.nanmean(area_vals)
        imputation_val_ellip = np.nanmean(ellipticity_vals)
        imputation_val_elon = np.nanmean(elongation_vals)
        
        
        #Impute missing ones
        if(len(failed_idxs) > 0): #If not empty
            for idx in failed_idxs:
                #Retrieve and impute with MEAN
                area_vals[idx] = imputation_val_area
                ellipticity_vals[idx] = imputation_val_ellip
                elongation_vals[idx] = imputation_val_elon

        


    end = timer()
    print("Time taken: {} seconds".format(np.around(end-start, 2)))
    print('\n')
    
    #How many values were imputed?
    
    
    print("# Imputed Values %: {}".format(np.around(100* len(failed_idxs)/len(closest_samples),2)))


    #Summary Stats for this node
    mu_area, sigma_area = np.mean(area_vals), np.std(area_vals)
    mu_ellipticity , sigma_ellipticity = np.mean(ellipticity_vals),np.std(ellipticity_vals)
    mu_elongation , sigma_elongation = np.mean(elongation_vals), np.std(elongation_vals)

    #Form tuple and return:
    summary_stats = [(mu_area, sigma_area), 
                     (mu_ellipticity , sigma_ellipticity), 
                     (mu_elongation , sigma_elongation)]

    return summary_stats



    
'''
Takes in a Stat_DICT

returns pooled_Area, pooled_ellip, pooled_elong


'''
def obtain_pooled_stats(stat_dict):
    #Pooled Area vals from ALL NODES
    pooled_areas = []
    pooled_ellipticities = []
    pooled_elongations =[]

    for k,v in stat_dict.items():
        #print("Node: {} \t Stats: {}".format(k,v))
        #Store BMU
        bmu = k
        #Store stats - i=Feature, j---Mean/SD from Tuple
        mu_area, sd_area = v[0][0], v[0][1]
        mu_ellipticity, sd_ellipticity = v[1][0], v[1][1]
        mu_elongation, sd_elongation = v[2][0], v[2][1]
        
        #add
        pooled_areas.append((mu_area, sd_area))
        pooled_ellipticities.append((mu_ellipticity, sd_ellipticity))
        pooled_elongations.append((mu_elongation, sd_elongation))
        
    return pooled_areas,pooled_ellipticities,pooled_elongations


'''

Takes in pooled Stats and plots mean lines with Error Bars


'''



def plot_stats(pooled_areas, pooled_ellipticities, pooled_elongations):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1 , figsize = (15,10), sharex=True)

    actual_linewidth = 1.6
    error_lwd = 0.8
    
    map_len = map_size[0]*map_size[1]

    idx = range(0,map_len)
    #Select subset
    areas = pooled_areas
    ellipticities = pooled_ellipticities
    elongations = pooled_elongations


    #Sort by MEAN IN-PLACE
    areas.sort(key = lambda tup: tup[0],reverse =True)
    ellipticities.sort(key = lambda tup: tup[0],reverse =True)
    elongations.sort(key = lambda tup: tup[0],reverse =True)




    #-----------Plot Area
    #ax1.set_xlabel('BMU')
    ax1.set_ylabel('Area')

    #Plotting data
    mu_area = [val[0] for val in areas]
    sd_area = [val[1] for val in areas]

    ax1.plot(mu_area,linewidth= actual_linewidth)
    ax1.errorbar(idx, mu_area, yerr = sd_area, fmt='-o', color='blue',linewidth=error_lwd)
    ax1.set_title('Mean Area vs. BMU (Descending Order)')

    #-----------Plot Elllipticities
    #ax2.set_xlabel('BMU')
    ax2.set_ylabel('Ellipticities')

    #Plotting data
    mu_ellip = [val[0] for val in ellipticities]
    sd_ellip = [val[1] for val in ellipticities]

    ax2.plot(mu_ellip,linewidth= actual_linewidth)
    ax2.errorbar(idx, mu_ellip, yerr = sd_ellip, fmt='-o',color='green',linewidth=error_lwd)
    ax2.set_title('Mean Ellipticities vs. BMU (Descending Order)')


    #-----------Plot Elllipticities
    ax3.set_xlabel('BMU')
    ax3.set_ylabel('Elongation')

    #Plotting data
    mu_elon = [val[0] for val in elongations]
    sd_elon = [val[1] for val in elongations]

    ax3.plot(mu_elon,linewidth= actual_linewidth)
    ax3.errorbar(idx, mu_elon, yerr = sd_elon, fmt='-o', color='red', linewidth=error_lwd)
    ax3.set_title('Mean Elongations vs. BMU (Descending Order)')


    plt.tight_layout()
    plt.show()



    
#---------------------------------------------------------ORIENTATION -----------------------------------


'''
Takes in a SOURCE Image from 'Closest Samples'
 and the appropriate SegmentationImage(sources) for that source (Non-Deblended Ideally)
 
RETURNS a Orientation
or NONE if Can't detect sources
'''

def find_orientation(data, sources):

    #sources is a SegmentationImage object
    if(sources==None):
        #print("Couldn't detect source!")
        return None
    
    #Find the Table of Sources
    cat = source_properties(data, sources)
    table = cat.to_table()
    #Format data and retain useful ones
    table['orientation'].info.format ='%.8g'
    
    #Get data
    orientation_vals = cat.orientation.value
    #Return Value
    orientation = np.mean(orientation_vals)
 
    return orientation



"""   

Plots image, source, Rotated IMG

 """
def plot_detected_sources_with_theta(data, deblended_sources):
    
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
        print(theta)
#         inertia = obj.inertia_tensor.value
#         print(inertia)
        orientation = obj.orientation.value
        print("Orientation: \t" + str(orientation))

        
        
        
        #Form and ADD
        aperture = EllipticalAperture(position, a , b, theta = theta)
        apertures.append(aperture)
        
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize =(20,10))
    #Plot Original Image
    ax1.imshow(data, cmap='gray' , norm = norm) #norm as Before
    ax1.set_title('Original Image')
    #Plot Segmented Image
    #Make color map
    cmap = segm_deblended.make_cmap(random_state= 123 )
    ax2.imshow(segm_deblended, cmap=cmap)
    ax2.set_title('Segmented Image')
        
    #----------Plot rotated image
    #Find orientation
    orientation = find_orientation(data, segm_deblended)
    rotated_im = ndimage.rotate(data, orientation)
    ax3.imshow(rotated_im, cmap= 'gray')
    ax3.set_title('Rotated Image')
    
    #Plot Apertures
    for aperture in apertures:
        aperture.plot(axes = ax1, lw =1.5, color ='white')
        aperture.plot(axes = ax2, lw =1.5, color ='white')

    plt.show()


'''
Pass in Rotated IMAGES, no need to reshape (each has 44x44 ish)

'''

def plot_rotated_samples(rotated_images, coords):
    #How many rotated images?
    num_samples = len(rotated_images)
    #Random Sample
    if(num_samples > 20):
        #Select only 20 randomly
        rotated_closest_img_list = sample(rotated_images, 20)
        num_samples = 20
    #Setup plot
    plt.figure(figsize=(20, 4))

    for i in range( 1, num_samples):
        ax = plt.subplot(1, num_samples, i)
        plt.imshow(rotated_closest_img_list[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #Filename
    filename = 'rotated_'+ str(coords[0]) + '_' + str(coords[1]) +'.png'
    img_savepath = os.path.join(current_path,'results/tmp/Images/',filename)
    plt.savefig(img_savepath)
    plt.show()


'''
Given a list of closest samples,
rotates each and every one by the orientation of the source

For situations where SE failed, we rotate by Mean orientation of all Samples(be careful!)
'''


def rotate_all_closest_samples(closest_samples):
    #Hold Orientation values
    angles = []
    #Hold rotated images
    rotated_samples = []
    failed_extractions = []
    num_failed = 0

    for xi in closest_samples:
        #Find deblended SOURCE with SE
        segm_non_deblended = get_non_deblended_sources(xi,  #The sample
                                                       nsigma = 3,
                                                       connected_points =  5)
        #Find Orientation
        orientation = find_orientation(xi, segm_non_deblended)
        #Check if we successfully detected a Source
        if(orientation):
            #Rotate and add rotated image data to collection
            rotated_samples.append(ndimage.rotate(xi, orientation))
            #Add the angle
            angles.append(orientation)
        else:
            #Deal with it later
            failed_extractions.append(xi)
            #Keep track
            num_failed +=1

    # Rotate now
    mean_angle = np.mean(angles)
    for failed_rotation in failed_extractions:
        #Rotate and add rotated image data to collection
        rotated_samples.append(ndimage.rotate(failed_rotation, mean_angle))
    
    
    print("Failed Extractions : {} %".format(100 * np.around((num_failed/len(rotated_samples)),2)))
    return rotated_samples
