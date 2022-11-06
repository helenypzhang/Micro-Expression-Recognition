#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Model
import numpy as np
import Training
# import Prepare as Prepare
import Validation


# In[2]:

'''

Changed Links for LINUX 

Uncomment to get actual links

also changed in prepare.py REMEMBER TO CHANGE
'''


# SAMM_dataset_path = 'C:/Users/mohamazim2/Downloads/SAMM/'
# SAMM_dataset_path = '/home/bsft19/mohamazim2/Windows/Downloads/SAMM/'
# SAMM_dataset_path = "E:/unzipped_cs229/SAMM/SAMM/Samm_with_fa/"
SAMM_dataset_path = "../SAMM/SAMM/"

# SMIC_dataset_path = '../SMIC/SMIC_all_cropped/HS/'
# SMIC_dataset_path = "../SMIC/SMIC/HS/"

# CK_dataset_path = '/home/bsft19/mohamazim2/Windows/Downloads/CK+/'
# CK_dataset_path = 'C:/Users/mohamazim2/Downloads/CK+_for_FA/'
# CK_dataset_path = "E:/unzipped_cs229/cs229/CK+/"
CK_dataset_path = '../CK+/'

# weightspathdir = '/home/bsft19/mohamazim2/Windows/Downloads/excel/'
# weightspathdir = '/home/bsft19/mohamazim2/Windows/Downloads/excel/'
#weightspathdir = '../SAMM/model_weights/' #folder to save weight; changed to aimed dataset;
weightspathdir = '../SAMM/weights_domain/'

# CK_dataset_path = "E:/unzipped_cs229/Face_alignment_on_ck+/CK+/"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(tf.__version__)
# print(tf.config.list_physical_devices())
model = 'DS'
MM = True
OF = True
fps = 100
# SAMM = Prepare.prepareSAMM(MM, OF, fps, SAMM_dataset_path)
# CK = Prepare.prepareCK(MM, OF, fps, CK_dataset_path)
# SMIC = Prepare.prepareSMIC(MM, OF, fps, SMIC_dataset_path)

print("begin training")
##for leave one sample out cross validation:
LOSOCV = Training.getLOSOCV('DS_domain', 'SAMM', 'CK+') #using 'DS'/'DS_domain'; 'SAMM'/'SMIC'/; 'CK+'/[]; #'DS_domain' 'CK+' ;   #'DS' [] ;

##for training:
# modelSAMM=Training.training(LOSOCV, weightspathdir)
# print("done training")

##for model parameters info:
# print(modelSAMM.summary())

##for evaluation:
Validation.validate2(LOSOCV, weightspathdir)

##for calculate FLOPs info:
# from flops_cal import *
# print_flops(modelSAMM)
