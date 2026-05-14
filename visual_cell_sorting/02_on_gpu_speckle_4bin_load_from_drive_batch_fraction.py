import rpyc
from docopt import docopt
import os

import threading as t
import sys, gc
import random
import math
import re
import time
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
from glob import glob
import pandas as pd
import scipy.ndimage
import warnings
import io
import base64, zlib
import cv2
from statistics import mode

from skimage.color import rgb2gray
from skimage.filters import sobel, threshold_multiotsu, threshold_otsu
from skimage.measure import regionprops, label

import cellpose
from cellpose import core, models, utils, plot
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)
from cellpose.io import logger_setup
logger_setup();

from fastai.vision.all import *
from fastai.callback.hook import *
#from fastai.utils.mem import *
from PIL import Image


class LinuxService(rpyc.Service):
    class exposed_ImageAnalysis(object):
        def __init__(self, callback):
        
            self.callback = rpyc.async_(callback)   # make the callback async
            return

        def exposed_finalize(self):
            #Save data as dataframes
            image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated_Q4', 'NCells_Activated_Q3', 'NCells_Activated_Q2', 'NCells_Activated_Q1', 'SegmentationTime', 'TotalTime'])            
            
            cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Nucleolar_Area', 'Nuclear_Area', 'Nucleolar_Ratio', 'UV_Status']) 
            image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
            cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
            return

        def exposed_run_pipeline_on_image(self):
            
            global image_number
            global prev_file

            image_list = glob.glob(os.path.join(img_save_dir, '*w1.TIF'))
            seg_file = max(image_list, key=os.path.getctime)
            if prev_file != seg_file:
                prev_file = seg_file
            else:
                image_list = glob.glob(os.path.join(img_save_dir, '*w1.TIF'))
                #seg_file = random.sample(image_list, 1)[0]
                seg_file = max(image_list, key=os.path.getctime)
            
            intens_file = seg_file[:-6] + 'Camera ' + channel  + '_ZStream.TIF'

            # Import image 
            intensity_img = skimage.io.imread(intens_file)
            intensity_img = np.max(intensity_img, 0)
            
            # Find the segmentation channel
            seg_img = skimage.io.imread(seg_file)
            
            # Copy file to another folder
            shutil.move(seg_file, os.path.join(img_move_dir, os.path.basename(seg_file)))
            shutil.move(intens_file, os.path.join(img_move_dir, os.path.basename(intens_file)))

            # Crop down to the center to make the image the expected size
            if input_image_height != image_height or input_image_width != image_width:
                start_x = input_image_width//2-(image_width//2)
                start_y = input_image_height//2-(image_height//2)
                img = img[start_y:start_y+image_height,start_x:start_x+image_width,:]

            #analyze
            output_mask = self.analyze_image(seg_img, intensity_img, intens_file)
            # Bring back up to the input size
            if input_image_height != image_height or input_image_width != image_width:
                start_x = input_image_width//2-(image_width//2)
                start_y = input_image_height//2-(image_height//2)
                output_mask = np.pad(output_mask, ((start_y,input_image_height-start_y-image_height), (start_x, input_image_width-start_x-image_width)), mode='constant', constant_values=0)

            #encode base64 and output
            if not output_mask.flags.c_contiguous:
                output_mask = output_mask.copy(order='C')
            
            mask_name = 'mask' + str(image_number) + '.png'
            mask_save_dir = os.path.join(mask_out_dir, mask_name)
            skimage.io.imsave(mask_save_dir, output_mask)

            return image_number

        def normalize_image(self, img, img_type='image', percentile=99.5):
            high = np.percentile(img, percentile)
            low = np.percentile(img, 100-percentile)

            img = np.minimum(high, img)
            img = np.maximum(low, img)

            img = (img - low) / (high - low)
            
            if img_type == 'image':
                img_norm = img
            else:
                img = skimage.img_as_ubyte(img)
                img_norm = np.stack([img, img, img], axis=-1)
            
            return img_norm

        def new_batch_pred_segmentation(self, batches, learn):
            outs = []
            learn.model.eval()
            with torch.no_grad():
                for b in batches:
                    outs.append(learn.model(b)) 
            
            if batches[0].shape[0] > 1 and batches[0].shape[0] != batches[-1].shape[0]:
                inp = torch.stack(batches[:-1])
                out = torch.stack(outs[:-1])
                dec = learn.dls.decode_batch((*tuplify(inp), *tuplify(out)), 10000)

                inp = torch.unsqueeze(batches[-1], 0)
                out = torch.unsqueeze(outs[-1], 0)
                dec2 = learn.dls.decode_batch((*tuplify(inp), *tuplify(out)), 10000)

                dec = dec + dec2

            else:
                inp = torch.stack(batches)
                out = torch.stack(outs)
                dec = learn.dls.decode_batch((*tuplify(inp), *tuplify(out)), 10000)

            return(dec)
            
        def new_batch_pred_classification(self, batches, learn):
            outs = []
            learn.model.eval()

            with torch.no_grad():
                for b in batches:
                    outs.append(learn.model(b)) 

            out = [pred.cpu().numpy().argmax(axis=1) for pred in outs]
            
            return(out)

        def analyze_image(self, segmentation_image, intensity_image, image_dir):

            # Catch warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                #mold image
                start_time = time.time()
                
                #normalize image for segmentation
                segmentation_image = self.normalize_image(segmentation_image, img_type='image', percentile=99) #99.5
                
                #normalize image for nucleolin
                intensity_image_norm = self.normalize_image(intensity_image, img_type='image', percentile=99.7) #99.5
                
                #perform segmentation
                masks, flows, styles, diams = model.eval(segmentation_image, diameter=90, normalize=False,
                                         cellprob_threshold=cellprob_threshold,
                                         flow_threshold=flow_threshold,
                                         channels=channels)
                masks = cellpose.utils.fill_holes_and_remove_small_masks(masks, min_size=1200)
                masks = cellpose.utils.remove_edge_masks(masks)
                rps = skimage.measure.regionprops(masks)

                n_cells = len(rps)
                print('Number of cells segmented: ' + str(n_cells))
                
                if n_cells > batch_size:
                    batch = [None]*batch_size
                else:
                    batch = [None]*n_cells
                
                batch_k = 0
                batches = []

                #iterate over masks in image
                cell_y_min = [0] * n_cells
                cell_y_max = [0] * n_cells
                cell_x_min = [0] * n_cells
                cell_x_max = [0] * n_cells
                cell_y_centroid = [0] * n_cells
                cell_x_centroid = [0] * n_cells
                crop_y_min = [0] * n_cells 
                crop_y_max = [0] * n_cells  
                crop_x_min = [0] * n_cells
                crop_x_max = [0] * n_cells
                nuclear_areas = [0] * n_cells       
                
                #fastai functions
                type_tfms = [PILImage.create]
                item_tfms = [Resize(224), ToTensor()]
                type_pipe = Pipeline(type_tfms)
                item_pipe = Pipeline(item_tfms)
                norm = Normalize.from_stats(*imagenet_stats)
                i2f = IntToFloatTensor()
                
                for i,prop in enumerate(rps):
                    cell_y_min[i] = prop.bbox[0]
                    cell_y_max[i] = prop.bbox[2]
                    crop_y_min[i] = np.maximum(0,cell_y_min[i]-expand_pixels)
                    crop_y_max[i] = np.minimum(segmentation_image.shape[0],cell_y_max[i]+expand_pixels)

                    cell_x_min[i] = prop.bbox[1]
                    cell_x_max[i] = prop.bbox[3]
                    crop_x_min[i] = np.maximum(0,cell_x_min[i]-expand_pixels)
                    crop_x_max[i] = np.minimum(segmentation_image.shape[1],cell_x_max[i]+expand_pixels)

                    cell_y_dim = cell_y_max[i] - cell_y_min[i]
                    cell_x_dim = cell_x_max[i] - cell_x_min[i]
                    crop_y_dim = crop_y_max[i] - crop_y_min[i]
                    crop_x_dim = crop_x_max[i] - crop_x_min[i]

                    cell_y_centroid[i] = np.int_((cell_y_min[i] + cell_y_max[i])/2)
                    cell_x_centroid[i] = np.int_((cell_x_min[i] + cell_x_max[i])/2)

                    crop = intensity_image_norm[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]
                    mask = masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]
                    mode_mask = mode(mask.flatten())
                    if mode_mask == 0:
                        mode_mask = np.unique(mask, return_counts=True)[0][1]
                    mask = np.where(mask == mode_mask, mask, 0)     

                    crop = crop*mask
                    y_pad = (crop_size - crop_y_dim)//2
                    y_odd = crop_y_dim % 2
                    x_pad = (crop_size - crop_x_dim)//2
                    x_odd = crop_x_dim % 2
                    crop = np.pad(crop, ((max(0,y_pad),max(0,y_pad + y_odd)), (max(0,x_pad),max(0,x_pad + x_odd))), mode='constant')[max(0,-y_pad):max(0,-y_pad)+crop_size, max(0,-x_pad):max(0,-x_pad)+crop_size]
                    
                    #normalize crop
                    norm_crop = self.normalize_image(crop, img_type='crop', percentile=99.7) #99.8
                    
                    #resize to 384x384
                    norm_crop = cv2.resize(norm_crop, dsize=(norm_crop.shape[0]*3, norm_crop.shape[1]*3), interpolation=cv2.INTER_CUBIC)
                    norm_crop = np.pad(norm_crop, pad_width=[(max(0,384-norm_crop.shape[0]),0),(0,max(0,384-norm_crop.shape[1])),(0,0)])
                    batch[batch_k] = item_pipe(type_pipe(norm_crop))
                    nuclear_areas[i] = np.sum(batch[batch_k][0].numpy()>0)
                    batch_k += 1
                    if batch_k == batch_size or i == n_cells - 1:
                        batches.append(torch.cat([norm(i2f(b.cuda())) for b in batch]))
                        if n_cells - batch_size*len(batches) > batch_size:
                            batch = [None] * batch_size
                        else:
                            batch = [None]* (n_cells - batch_size*len(batches))
                        batch_k = 0

                n_cells_with_nucleoli = 0

                
                # Perform fastai predictions of speckles on gpu
                if n_cells > 0 and not random_activation:
                    nuclear_areas_with_nucleoli = nuclear_areas #np.array(nuclear_areas)[predictions == 1]
                    n_cells_with_nucleoli = len(nuclear_areas_with_nucleoli)
                    concat_batch = torch.cat(batches)

                    if n_cells_with_nucleoli > 0:
                        with learn_seg.no_bar():
                            out = self.new_batch_pred_segmentation(batches, learn_seg)
                            preds = [m.cpu().numpy().argmax(axis=0) for batch in out for m in batch[1]]
                        
                        nucleolar_areas = [0] * n_cells_with_nucleoli

                        for j, pred in enumerate(preds):
                            labels = skimage.measure.label(pred)
                            df_props = pd.DataFrame(skimage.measure.regionprops_table(labels, properties=['area']))
                            nuc_filter = list(df_props[df_props['area'] < area_cutoff].index)
                            #print(df_props)
                            if nuc_filter:
                                for lab in nuc_filter:
                                    labels[labels==lab+1] = 0
                            nucleolar_areas[j] = np.sum(labels>0)
                        
                        print('Number of cells with nucleoli: ' + str(n_cells_with_nucleoli))
                        print(pd.Series(nucleolar_areas).describe())
                    
                
                if random_activation:
                    predictions = [0] * n_cells 

                #construct output mask and save information about objects
                uv_q1_mask = np.zeros((image_height, image_width), dtype='bool')
                uv_q2_mask = np.zeros((image_height, image_width), dtype='bool')
                uv_q3_mask = np.zeros((image_height, image_width), dtype='bool')
                uv_q4_mask = np.zeros((image_height, image_width), dtype='bool') 
                
                n_activated_cells_q1 = 0
                n_activated_cells_q2 = 0
                n_activated_cells_q3 = 0
                n_activated_cells_q4 = 0
                counter = 0
                
                for i in range(n_cells):
                    nucleolin_pred = 0
                    if random_activation:
                        nucleolin_pred = random.choice((1,2,3,4))
                    else:
                        nuc_area = nucleolar_areas[i]
                        nuclear_area = nuclear_areas_with_nucleoli[i]
                        nucleolar_ratio = nuc_area / nuclear_area
                        if nuc_area > SC35_cutoff:
                            if nucleolar_ratio >= nucleolar_ratio_cutoff_high + nucleolar_ratio_cutoff_offset:
                                nucleolin_pred = 4
                            elif (nucleolar_ratio >= nucleolar_ratio_cutoff_mid + nucleolar_ratio_cutoff_offset) and (nucleolar_ratio < nucleolar_ratio_cutoff_high):
                                nucleolin_pred = 3
                            elif (nucleolar_ratio >= nucleolar_ratio_cutoff_low + nucleolar_ratio_cutoff_offset) and (nucleolar_ratio < nucleolar_ratio_cutoff_mid):
                                nucleolin_pred = 2
                            elif nucleolar_ratio <= nucleolar_ratio_cutoff_low - nucleolar_ratio_cutoff_offset:
                                nucleolin_pred = 1

                    #save object information
                    global image_number
                    global celldata
                    global object_number
                    celldata[object_number] = [
                        image_number, 
                        image_dir, 
                        object_number, 
                        i, 
                        nuc_area,
                        nuclear_area,
                        nucleolar_ratio,
                        None
                    ]
                    object_number += 1
                    
                    nuc_label = 'NA'
                    
                    if nucleolin_pred == 4:
                        uv_q4_mask[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]==i+1
                        nuc_label = 'Q4'
                        n_activated_cells_q4 += 1
                    elif nucleolin_pred == 3:
                        uv_q3_mask[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]==i+1
                        nuc_label = 'Q3'
                        n_activated_cells_q3 += 1
                    elif nucleolin_pred == 2:
                        uv_q2_mask[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]==i+1
                        nuc_label = 'Q2'
                        n_activated_cells_q2 += 1
                    elif nucleolin_pred == 1:
                        uv_q1_mask[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]==i+1
                        nuc_label = 'Q1'
                        n_activated_cells_q1 += 1

                    celldata[(object_number-1)][7] = nuc_label

                output_mask = uv_q1_mask.astype(np.uint8)
                if n_cells > 0:
                    output_mask[uv_q4_mask] = 4
                    output_mask[uv_q3_mask] = 3
                    output_mask[uv_q2_mask] = 2

                # Save image information
                end_time2 = time.time()
                segmentation_time = total_time = end_time2 - start_time
                global imagedata
                imagedata[image_number] = [image_number, image_dir, n_cells, n_activated_cells_q4, n_activated_cells_q3, n_activated_cells_q2, n_activated_cells_q1, segmentation_time, total_time]
                image_number += 1
                if image_number % save_every == 0:
                    # Clear cache
                    os.system(shell_script)
                    #Save data as dataframes
                    image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated_Q4', 'NCells_Activated_Q3', 'NCells_Activated_Q2', 'NCells_Activated_Q1', 'SegmentationTime', 'TotalTime'])
                    cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Nucleolar_Area', 'Nuclear_Area', 'Nucleolar_Ratio', 'UV_Status']) 
                    image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
                    cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
                print('Image number: ' + str(image_number))
                print('Total processing time in ms: ' + str(total_time*1000))
                print('Cutoffs: ' + str(round(nucleolar_ratio_cutoff_low, 3)) + ', ' + str(round(nucleolar_ratio_cutoff_mid, 3)) + ', ' + str(round(nucleolar_ratio_cutoff_high, 3)))
                print('Number of cells (Q1, Q2, Q3, Q4): ' + str(n_activated_cells_q1) + ', ' + str(n_activated_cells_q2) + ', ' + str(n_activated_cells_q3) + ', ' + str(n_activated_cells_q4))
            
            return output_mask

if __name__ == "__main__":

    random.seed()
    args = docopt(__doc__, version='1.0')
    
    output_dir = args['<output>'] 
    
    #Set global vars
    model = cellpose.models.Cellpose(gpu=use_GPU, model_type='nuclei')
    channels = [0,0]
    
    Flow_Threshold = 0.2 #@param {type:"slider", min:0.1, max:1.1, step:0.1}
    flow_threshold=Flow_Threshold
    Cell_Probability_Threshold=0.5 #@param {type:"slider", min:-6, max:6, step:1}
    cellprob_threshold=Cell_Probability_Threshold
    
    save_every = 100
    area_cutoff = 15
    SC35_cutoff = 150
    batch_size = 64
    expand_pixels = 2
    boundary_size = 3
    input_image_height = int(args['--height'])
    input_image_width = int(args['--width'])
    image_height = 1536
    image_width = 1739
    crop_size = 128
    celldata = {}
    image_number = 0
    imagedata = {}
    object_number = 0
    use_single_channel = False #int(args['--use-single-channel']) if args['--use-single-channel'] is not '0' else False
    intensity_channel = 1 #int(args['--intensity-channel']) if args['--intensity-channel'] is not '0' else 2
    segmentation_channel = 2 #int(args['--segmentation-channel']) if args['--segmentation-channel'] is not '0' else 1
    random_activation = bool(args['--random'])
    no_img_save = False #bool(args['--no-images'])
    
    img_save_dir = '/mnt/nas2/2023/Hyeon-Jin/04-28-23_F1-E15.25-SC35-plate4-ratio-C'
    img_move_dir = img_save_dir + '_images'
    mask_out_dir = img_save_dir + '_masks'
    prev_file = ''
    
    channel = 'NIR'
    nucleolar_ratio_cutoff_low = 0.0951 # 0.1180, 0.1049, 0.0956, 0.0951
    nucleolar_ratio_cutoff_mid = 0.1164 # 0.1392, 0.1246, 0.1165, 0.1164
    nucleolar_ratio_cutoff_high = 0.1374 # 0.1595, 0.1438, 0.1364, 0.1374
    nucleolar_ratio_cutoff_offset = 0.0001

    if not os.path.exists(img_move_dir):
        os.makedirs(img_move_dir)

    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    segmentation_model_path = '/home/hyeonjin/Projects/fastai_models/compartment_segmentation/export.pkl'

    def label_func(x): return x.parent.parent.parent/'annotations'/(x.stem+'.png')
    
    learn_seg = load_learner(segmentation_model_path, cpu=False)

    #Initialize pytorch model
    defaults.device = 'cuda'
    device = torch.device(defaults.device)
    run_server = True    
    
    # For clearing cache
    password = ''
    command = 'sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'

    # formatting the sudo password and the command
    shell_script = f"echo {password} | sudo -S {command}"

    if run_server:
        print('Pytorch is using GPU!')

        #Start rpyc server
        print('Ready to accept images!')
        from rpyc.utils.server import ThreadedServer
        ThreadedServer(LinuxService, port = 18871).start()
    else:
        service = LinuxService.exposed_ImageAnalysis()
        indir = Path('../457')
        outdir = Path('/tmp/457')
        outdir.mkdir(exist_ok=True)
        service.analyze_image(None, np.load(indir.joinpath('image.npy')), outdir)
        service.exposed_finalize()

