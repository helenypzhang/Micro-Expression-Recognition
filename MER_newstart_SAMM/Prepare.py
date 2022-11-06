#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
# import imageio
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
# from keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import sys
import pandas
import Utility as u
from tensorflow.keras.utils import to_categorical
# In[2]:

# K.set_image_data_format('channels_first')
image_rows, image_columns, image_depth = 64, 64, 18
flow_rows, flow_columns, flow_depth = 144, 120, 16


# In[20]:


def prepareSAMM(MM, OF, fps, SAMM_dataset_path):
    
    SAMM = []

    dataSheet = pandas.read_excel(io='../SAMM/xcel.xlsx', usecols='B, D:F, G, J', index_col=0, engine='openpyxl')
    # dataSheet = pandas.read_excel(io= 'C:/Users/mohamazim2/Downloads/excel/xcel.xlsx', usecols= 'B, D:F, G, J', index_col = 0)
    # dataSheet = pandas.read_excel(io="/home/bsft19/mohamazim2/Windows/Downloads/excel/xcel.xlsx", usecols= 'B, D:F, G, J', index_col = 0)
    # dataSheet = pandas.read_excel(io = "E:/unzipped_cs229/SAMM/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx" ,usecols= 'B, D:F, G, J', index_col = 0)
    SAMM_list = []
    SAMM_labels = []
    SAMM_flow_list = []
    directorylisting = os.listdir(SAMM_dataset_path)
    SAMM_subject_boundary = []
    SAMM_subject_boundary.append(0)
    print(dataSheet)
    for subject in directorylisting:

        boundary = SAMM_subject_boundary[-1]

        print("SAMM current subject: ", subject)

        for sample in os.listdir(SAMM_dataset_path + subject + '/'):

            apex = dataSheet.loc[sample, 'Apex Frame']

            emotion = dataSheet.loc[sample, 'Estimated Emotion']

            onset = dataSheet.loc[sample, 'Onset Frame']

            offset = dataSheet.loc[sample, 'Offset Frame']

            frames = []

            frame_count = 18

            height = 640

            width = 960

            # for face alignment

            # height = 224
            # width = 224


            video_tensor=np.zeros((frame_count,height,width,3),dtype='float')


            if (dataSheet.loc[sample, 'Duration'] >= image_depth and emotion != 'Other'):

                label = 0

                if(emotion == 'Happiness'):
                    label = 1
                elif(emotion == 'Surprise'):
                    label = 2

                count = 0

                start = 0

                end = 0

                if(apex - onset > (image_depth / 2) and offset - apex >= (image_depth / 2)):

                    start = apex - (image_depth / 2)

                    end = apex + (image_depth / 2)

                elif(apex - onset <= (image_depth / 2)):

                    start = onset

                    end = onset + image_depth - 1

                elif(offset - apex < (image_depth / 2)):

                    start = offset - image_depth

                    end = offset

                for image in os.listdir(SAMM_dataset_path + subject + '/' + sample + '/'):

                    imagecode = int(image[-4 - len(str(apex)):-4])

                    if(imagecode >= start and imagecode < end):
                        image_path = SAMM_dataset_path  + subject + '/' + sample + '/' + image
                        img = cv2.imread(image_path)
                        frames.append(img)
                        temp = img[:640, : , :]
                        video_tensor[count] = temp

                        count = count + 1

                if(OF == True):

                    frames_flow = []
                    for frame in frames:
                        frame_resize = cv2.resize(frame, (flow_columns, flow_rows), interpolation = cv2.INTER_AREA)
                        frame_flow = np.float32(frame_resize)
                        frames_flow.append(frame_flow)

                    SAMM_flow = u.opticalFlow(2, frames_flow)
                    SAMM_flow = np.asarray(SAMM_flow)
                    SAMM_flowarray = np.rollaxis(np.rollaxis(SAMM_flow, 2, 0), 2, 0)
                    SAMM_flow_list.append(SAMM_flowarray)

                if(MM == True):
                    frames = u.magnify_video(video_tensor, fps, 0.4, 3, image_rows, image_columns, levels=3, amplification=20)
                else:
                    resized_frames = []
                    for frame in frames:
                        imageresize = cv2.resize(frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                        resized_frames.append(grayimage)
                    frames = resized_frames

                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                SAMM_list.append(videoarray)
                SAMM_labels.append(label)
                boundary = boundary + 1

            else:
                if(dataSheet.loc[sample, 'Duration'] < image_depth):
                    print('Unqualified sample' + sample + ' Duration:' + str(dataSheet.loc[sample, 'Duration']))
                else:
                    print('Unqualified sample' + sample + ' Emotion: Other')

        if(SAMM_subject_boundary[-1] != boundary):
            SAMM_subject_boundary.append(boundary)

    SAMMsamples = len(SAMM_list)
    SAMM_labels = to_categorical(SAMM_labels, 3)
    SAMM_data = [SAMM_list,SAMM_labels]
    (SAMMframes,SAMM_labels) = (SAMM_data[0], SAMM_data[1])
    SAMM_set = np.zeros((SAMMsamples, 1, image_rows, image_columns, image_depth))
    SAMM_flow_set = np.zeros((SAMMsamples, 1, flow_rows, flow_columns, flow_depth))

    for h in range(SAMMsamples):
        SAMM_set[h][0][:][:][:] = SAMMframes[h]
        SAMM_flow_set[h][0][:][:][:] = SAMM_flow_list[h]
        

    SAMM_set = SAMM_set.astype('float32')
    SAMM_set -= np.mean(SAMM_set)
    SAMM_set /= np.max(SAMM_set)
    
    SAMM.append(SAMM_set)
    SAMM.append(SAMM_labels)
    SAMM.append(SAMM_subject_boundary)
    if(SAMM_flow_list):
        SAMM.append(SAMM_flow_set)
        # np.save('C:/Users/mohamazim2/Downloads/excel/SAMM_flow.npy', SAMM_flow_set)
        # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_flow.npy", SAMM_flow_set)
        # np.save("/home/bsft19/mohamazim2/Windows/Downloads/excel/SAMM_flow.npy", SAMM_flow_set)
        np.save("../SAMM/results/SAMM_flow.npy", SAMM_flow_set)

    # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_set.npy", SAMM_set)
    # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_labels.npy", SAMM_labels)
    # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_subject_boundary.npy", SAMM_subject_boundary)

    np.save("../SAMM/results/SAMM_set.npy", SAMM_set)
    np.save("../SAMM/results/SAMM_labels.npy", SAMM_labels)
    np.save("../SAMM/results/SAMM_subject_boundary.npy", SAMM_subject_boundary)

    return SAMM


# In[7]:


def prepareCK(MM, OF, fps, CK_dataset_path):
    CK_list = []
    CK_labels = []
    CK_flow_list = []
    CK = []
    # image_path = CK_dataset_path +  'cohn-kanade-images/'

    image_path = CK_dataset_path + 'Images/'
    # label_path = CK_dataset_path + 'Emotion/'
    label_path = CK_dataset_path +  'Emotions/'

    CK_dict = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2}
    video_tensor=np.zeros((18,480,640,3),dtype='float')
    # video_tensor = np.zeros((18,224,224,3),dtype='float')
    print("DSfds")
    print(os.listdir(label_path))
    for subject in os.listdir(label_path):
        subject_label_path = os.path.join(label_path, subject)
        subject_image_path = os.path.join(image_path, subject)
        print("CK+ current subject: ", subject)
        for test in os.listdir(subject_label_path):
            
            frames = []
            test_label_path = subject_label_path + '/' + test
            print("==============================")
            print(test_label_path)
            test_image_path = subject_image_path + '/' + test
            print(test_image_path)
            if os.listdir(test_label_path):

                with open (os.path.join(test_label_path, os.listdir(test_label_path)[-1]), 'r') as rl:
                    label = rl.readline()
                    label = int(float(label.strip('\n')))

                if label != 0:
                    CK_labels.append(CK_dict[label])

                    image_dir = os.listdir(test_image_path)

                    apex_num = int(len(image_dir) / 2)

                    if (apex_num < 18):
                        if(apex_num >= 12):
                            mul = 2
                        else:
                            mul = round(18 / apex_num)

                        frameNum = 0
                        tensorLayer = 0

                        while (tensorLayer < 18):
                            count = 0
                            while(count < mul and tensorLayer < 18):
                                image = cv2.imread(test_image_path + '/' + image_dir[apex_num - frameNum])
                                frames.append(image)
                                temp = image[:480, :640, :]
                                video_tensor[17 - tensorLayer] = temp
                                
                                
                                count = count + 1
                                tensorLayer = tensorLayer + 1
                            frameNum = frameNum + 1

                    else:
                        for f in range(18):
                            image = cv2.imread(test_image_path + '/' + image_dir[apex_num - f])
                            frames.append(image)
                            temp = image[:480, :640, :]
                            video_tensor[17 - f] = temp
                            
                    
                    if(OF == True):
                        frames = reversed(frames)
                        frames_flow = []
                        for frame in frames:
                            frame_resize = cv2.resize(frame, (flow_columns, flow_rows), interpolation = cv2.INTER_AREA)
                            frame_flow = np.float32(frame_resize)
                            frames_flow.append(frame_flow)
                    
                        CK_flow = u.opticalFlow(2, frames_flow)

                        CK_flow = np.asarray(CK_flow)
                        CK_flowarray = np.rollaxis(np.rollaxis(CK_flow, 2, 0), 2, 0)
                        CK_flow_list.append(CK_flowarray)
                        
                    if(MM == True):
                        frames = u.magnify_video(video_tensor, fps, 0.4, 3, image_rows, image_columns, levels=3, amplification=20)
                    else:
                        resized_frames = []
                        for frame in frames:
                            imageresize = cv2.resize(frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                            grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                            resized_frames.append(grayimage)
                        frames = resized_frames
                        
                    videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                    CK_list.append(videoarray)
                    
    CK_labels = np.asarray(CK_labels)
    CK_labels = to_categorical(CK_labels, 3)

    CKsamples = len(CK_list)
    CK_data = [CK_list, CK_labels]
    (CKframes, CK_labels) = (CK_data[0], CK_data[1])
    CK_set = np.zeros((CKsamples, 1, image_rows, image_columns, image_depth))
    CK_flow_set = np.zeros((CKsamples, 1, flow_rows, flow_columns, flow_depth))

    for h in range(CKsamples):
        CK_set[h][0][:][:][:] = CKframes[h]
        CK_flow_set[h][0][:][:][:] = CK_flow_list[h]

    CK_set = CK_set.astype('float32')
    CK_set -= np.mean(CK_set)
    CK_set /= np.max(CK_set)

#     CK_flow_set = CK_flow_set.astype('float32')
#     CK_flow_set -= np.mean(CK_flow_set)
#     CK_flow_set /= np.max(CK_flow_set)

    CK.append(CK_set)
    CK.append(CK_labels)
    if(CK_flow_list):
        CK.append(CK_flow_set)
        # "/home/bsft19/mohamazim2/Windows/Downloads/excel/SAMM_set.npy
        # np.save('/home/bsft19/mohamazim2/Windows/Downloads/excel/CK+/CK_flow.npy', CK_flow_set)
        # np.save("C:/Users/mohamazim2/Downloads/CK+_for_FA/fa_res/CK_flow.npy", CK_flow_set)
        # np.save("E:/unzipped_cs229/Face_alignment_on_ck+/results/CK_flow.npy", CK_flow_set)
        np.save("../CK+/results/CK_flow.npy", CK_flow_set)

    # np.save("E:/unzipped_cs229/Face_alignment_on_ck+/results/CK_set.npy", CK_set)
    # np.save("E:/unzipped_cs229/Face_alignment_on_ck+/results/CK_labels.npy", CK_labels)

    np.save("../CK+/results/CK_set.npy", CK_set)
    np.save("../CK+/results/CK_labels.npy", CK_labels)

    return CK


# In[8]:


def prepareSMIC(MM, OF, fps, SMIC_dataset_path):
    SMIC = []
    SMIC_list = []
    SMIC_labels = []
    SMIC_flow_list = []
    directorylisting = os.listdir(SMIC_dataset_path)
    subject_boundary = []
    
    video_tensor = np.zeros((18, 144, 120, 3), dtype='float')

    subject_boundary.append(0)

    for subject_path in directorylisting:
        
        print("SMIC current subject: ", subject_path)

        boundary = subject_boundary[-1]
        path = []

        path.append(SMIC_dataset_path + subject_path + '/micro/negative/') 
        path.append(SMIC_dataset_path + subject_path + '/micro/positive/')
        path.append(SMIC_dataset_path + subject_path + '/micro/surprise/')

        # for each label in one subject

        for label in range(3):

            for example in os.listdir(path[label]):

                example_path = path[label] + example

                frames = []
                framelisting = os.listdir(example_path)
                framerange = 18
                i = 0

                if(len(framelisting) < framerange):
                    framerange = len(framelisting)

                for frame in range(framerange):
                    imagepath = example_path + "/" + framelisting[frame]
                    image = cv2.imread(imagepath)
                    temp = image[:144, :120, :]
                    video_tensor[i] = temp
                    i = i + 1
                    frames.append(image)

                for duplicate in range(18 - framerange):
                    imagepath = example_path + "/" + framelisting[framerange - 1]
                    image = cv2.imread(imagepath)
                    temp = image[:144, :120, :]
                    video_tensor[i] = temp
                    i = i + 1
                    frames.append(image)
                
                if(OF == True):

                    frames_flow = []

                    for frame in frames:
                        frame_resize = cv2.resize(frame, (flow_columns, flow_rows), interpolation = cv2.INTER_AREA)
                        frame_flow = np.float32(frame_resize)
                        frames_flow.append(frame_flow)
                    
                    SMIC_flow = u.opticalFlow(2, frames_flow)

                    SMIC_flow = np.asarray(SMIC_flow)
                    SMIC_flowarray = np.rollaxis(np.rollaxis(SMIC_flow, 2, 0), 2, 0)
                    SMIC_flow_list.append(SMIC_flowarray)
                
                if(MM == True):
                    frames = u.magnify_video(video_tensor, fps, 0.4, 3, image_rows, image_columns, levels=3, amplification=20)
                else:
                    resized_frames = []
                    
                    for frame in frames:
                        imageresize = cv2.resize(frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                        resized_frames.append(grayimage)
                    frames = resized_frames
                    

                frames = np.asarray(frames)
                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                SMIC_list.append(videoarray)
                SMIC_labels.append(label)
                boundary = boundary + 1

        subject_boundary.append(boundary)
    
    SMIC_labels = np.asarray(SMIC_labels)
    SMIC_labels =to_categorical(SMIC_labels, 3)

    SMICsamples = len(SMIC_list)
    SMIC_data = [SMIC_list, SMIC_labels]
    (SMICframes, SMIC_labels) = (SMIC_data[0], SMIC_data[1])
    SMIC_set = np.zeros((SMICsamples, 1, image_rows, image_columns, image_depth))
    SMIC_flow_set = np.zeros((SMICsamples, 1, flow_rows, flow_columns, flow_depth))

    for h in range(SMICsamples):
        SMIC_set[h][0][:][:][:] = SMICframes[h]
        SMIC_flow_set[h][0][:][:][:] = SMIC_flow_list[h]

    SMIC_set = SMIC_set.astype('float32')
    SMIC_set -= np.mean(SMIC_set)
    SMIC_set /= np.max(SMIC_set)
    
    SMIC.append(SMIC_set)
    SMIC.append(SMIC_labels)
    SMIC.append(subject_boundary)
    if(SMIC_flow_list):
        SMIC.append(SMIC_flow_set)
        # np.save('SMIC/SMIC_flow.npy', SMIC_flow_set)
        np.save("../SMIC/results/SMIC_flow.npy", SMIC_flow_set)

    # np.save('SMIC/SMIC_set.npy', SMIC_set)
    # np.save('SMIC/SMIC_labels.npy', SMIC_labels)
    # np.save('SMIC/SMIC_subject_boundary.npy', subject_boundary)

    np.save('../SMIC/results/SMIC_set.npy', SMIC_set)
    np.save('../SMIC/results/SMIC_labels.npy', SMIC_labels)
    np.save('../SMIC/results/SMIC_subject_boundary.npy', subject_boundary)

    return SMIC


# def prepareSAMMtest(MM, OF, fps, SAMM_dataset_path):
    
#     SAMM = []

#     dataSheet = pandas.read_excel(io='../SAMM/xcel.xlsx', usecols='B, D:F, G, J', index_col=0, engine='openpyxl')
#     # dataSheet = pandas.read_excel(io= 'C:/Users/mohamazim2/Downloads/excel/xcel.xlsx', usecols= 'B, D:F, G, J', index_col = 0)
#     # dataSheet = pandas.read_excel(io="/home/bsft19/mohamazim2/Windows/Downloads/excel/xcel.xlsx", usecols= 'B, D:F, G, J', index_col = 0)
#     # dataSheet = pandas.read_excel(io = "E:/unzipped_cs229/SAMM/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx" ,usecols= 'B, D:F, G, J', index_col = 0)
#     SAMM_list = []
#     SAMM_labels = []
#     SAMM_flow_list = []
#     directorylisting = os.listdir(SAMM_dataset_path)
#     SAMM_subject_boundary = []
#     SAMM_subject_boundary.append(0)
#     print(dataSheet)
#     for subject in directorylisting:

#         boundary = SAMM_subject_boundary[-1]

#         print("SAMM current subject: ", subject)

#         for sample in os.listdir(SAMM_dataset_path + subject + '/'):

#             apex = dataSheet.loc[sample, 'Apex Frame']

#             emotion = dataSheet.loc[sample, 'Estimated Emotion']

#             onset = dataSheet.loc[sample, 'Onset Frame']

#             offset = dataSheet.loc[sample, 'Offset Frame']

#             frames = []

#             frame_count = 18

#             height = 640

#             width = 960

#             # for face alignment

#             # height = 224
#             # width = 224


#             video_tensor=np.zeros((frame_count,height,width,3),dtype='float')


#             if (dataSheet.loc[sample, 'Duration'] >= image_depth and emotion != 'Other'):

#                 label = 0

#                 if(emotion == 'Happiness'):
#                     label = 1
#                 elif(emotion == 'Surprise'):
#                     label = 2

#                 count = 0

#                 start = 0

#                 end = 0

#                 if(apex - onset > (image_depth / 2) and offset - apex >= (image_depth / 2)):

#                     start = apex - (image_depth / 2)

#                     end = apex + (image_depth / 2)

#                 elif(apex - onset <= (image_depth / 2)):

#                     start = onset

#                     end = onset + image_depth - 1

#                 elif(offset - apex < (image_depth / 2)):

#                     start = offset - image_depth

#                     end = offset

#                 for image in os.listdir(SAMM_dataset_path + subject + '/' + sample + '/'):

#                     imagecode = int(image[-4 - len(str(apex)):-4])

#                     if(imagecode >= start and imagecode < end):
#                         image_path = SAMM_dataset_path  + subject + '/' + sample + '/' + image
#                         img = cv2.imread(image_path)
#                         frames.append(img)
#                         temp = img[:640, : , :]
#                         video_tensor[count] = temp

#                         count = count + 1

#                 if(OF == True):

#                     frames_flow = []
#                     for frame in frames:
#                         frame_resize = cv2.resize(frame, (flow_columns, flow_rows), interpolation = cv2.INTER_AREA)
#                         frame_flow = np.float32(frame_resize)
#                         frames_flow.append(frame_flow)

#                     SAMM_flow = u.opticalFlow(2, frames_flow)
#                     SAMM_flow = np.asarray(SAMM_flow)
#                     SAMM_flowarray = np.rollaxis(np.rollaxis(SAMM_flow, 2, 0), 2, 0)
#                     SAMM_flow_list.append(SAMM_flowarray)

#                 if(MM == True):
#                     frames = u.magnify_video(video_tensor, fps, 0.4, 3, image_rows, image_columns, levels=3, amplification=20)
#                 else:
#                     resized_frames = []
#                     for frame in frames:
#                         imageresize = cv2.resize(frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
#                         grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
#                         resized_frames.append(grayimage)
#                     frames = resized_frames

#                 videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
#                 SAMM_list.append(videoarray)
#                 SAMM_labels.append(label)
#                 boundary = boundary + 1

#             else:
#                 if(dataSheet.loc[sample, 'Duration'] < image_depth):
#                     print('Unqualified sample' + sample + ' Duration:' + str(dataSheet.loc[sample, 'Duration']))
#                 else:
#                     print('Unqualified sample' + sample + ' Emotion: Other')

#         if(SAMM_subject_boundary[-1] != boundary):
#             SAMM_subject_boundary.append(boundary)

#     SAMMsamples = len(SAMM_list)
#     SAMM_labels = to_categorical(SAMM_labels, 3)
#     SAMM_data = [SAMM_list,SAMM_labels]
#     (SAMMframes,SAMM_labels) = (SAMM_data[0], SAMM_data[1])
#     SAMM_set = np.zeros((SAMMsamples, 1, image_rows, image_columns, image_depth))
#     SAMM_flow_set = np.zeros((SAMMsamples, 1, flow_rows, flow_columns, flow_depth))

#     for h in range(SAMMsamples):
#         SAMM_set[h][0][:][:][:] = SAMMframes[h]
#         SAMM_flow_set[h][0][:][:][:] = SAMM_flow_list[h]
        

#     SAMM_set = SAMM_set.astype('float32')
#     SAMM_set -= np.mean(SAMM_set)
#     SAMM_set /= np.max(SAMM_set)
    
#     SAMM.append(SAMM_set)
#     SAMM.append(SAMM_labels)
#     SAMM.append(SAMM_subject_boundary)
#     # if(SAMM_flow_list):
#     #     SAMM.append(SAMM_flow_set)
#     #     # np.save('C:/Users/mohamazim2/Downloads/excel/SAMM_flow.npy', SAMM_flow_set)
#     #     # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_flow.npy", SAMM_flow_set)
#     #     # np.save("/home/bsft19/mohamazim2/Windows/Downloads/excel/SAMM_flow.npy", SAMM_flow_set)
#     #     np.save("../SAMM/results/SAMM_flow.npy", SAMM_flow_set)

#     # # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_set.npy", SAMM_set)
#     # # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_labels.npy", SAMM_labels)
#     # # np.save("E:/unzipped_cs229/SAMM/SAMM/results/SAMM_subject_boundary.npy", SAMM_subject_boundary)

#     # np.save("../SAMM/results/SAMM_set.npy", SAMM_set)
#     # np.save("../SAMM/results/SAMM_labels.npy", SAMM_labels)
#     # np.save("../SAMM/results/SAMM_subject_boundary.npy", SAMM_subject_boundary)

#     return SAMM
