# inference.py Jaume Colom - jaume.colom@tum.de
# This script runs inference on an trained model.
# It loads the model from the disk with a Saver, 
# computes the results for the given INPUT observations 
# and stores them to a .csv file.

# Necessary imports
import datetime
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
import os
import time

tp_path = os.path.join('/home/usuario/project/align3d', 'tp_utils')
sys.path.insert(0, tp_path)

import provider
import copy
import models.tp8 as MODEL_tp8
from config import load_config, configGlobal, save_config

from argparse import Namespace
import json
from pandas import DataFrame

from pandas import read_csv

def run_inference(configs, ids, ppath):
    """  """

    print(configs)
    load_config(configs[0])

    cfg = configGlobal
    MODEL = MODEL_tp8
    eval_epoch=configs[1]

    VAL_INDICES = ids

    with tf.Graph().as_default():
        # Define model on the device
        with tf.device('/gpu:' + str(cfg.gpu_index)):
            pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles = MODEL.placeholder_inputs(cfg.training.batch_size, cfg.model.num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            
            # Get model and loss
            end_points = MODEL.get_model(pcs1, pcs2, is_training_pl)
            loss = MODEL.get_loss(pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles, end_points)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=1000)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        sess = tf.Session(config=config)
        
        merged = tf.summary.merge_all() # Necessary to run


        # Init variables
        init = tf.global_variables_initializer()
        
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})
        
        ops = {'pcs1': pcs1, # INPUT
            'pcs2': pcs2, # INPUT
            'translations': translations, # GT 
            'rel_angles': rel_angles,     # gt
            'is_training_pl': is_training_pl, # What is it for?
            'pred_translations': end_points['pred_translations'], 
            'pred_remaining_angle_logits': end_points['pred_remaining_angle_logits'], 
            'pc1centers': pc1centers,     # GT
            'pc2centers': pc2centers,     # GT
            'pc1angles': pc1angles,       # GT
            'pc2angles': pc2angles,       # GT
            'pred_s1_pc1centers': end_points['pred_s1_pc1centers'], 
            'pred_s1_pc2centers': end_points['pred_s1_pc2centers'], 
            'pred_s2_pc1centers': end_points['pred_s2_pc1centers'], 
            'pred_s2_pc2centers': end_points['pred_s2_pc2centers'], 
            'pred_pc1angle_logits': end_points['pred_pc1angle_logits'], 
            'pred_pc2angle_logits': end_points['pred_pc2angle_logits'], 
            'loss': loss,                 # COMPUTED
            'merged': merged}             # WHAT IS THIS??
            #'train_op': train_op, 
            #'step': batch}                # INFO

        
        # Load existing model!
        model_to_load = cfg.logging.logdir
        assert os.path.isfile(f'{model_to_load}/model-{eval_epoch}.index'), f'{model_to_load}/model-{eval_epoch}.index'
        saver.restore(sess, f'{model_to_load}/model-{eval_epoch}')
        
        start_epoch = int(eval_epoch)
        epoch = start_epoch
                
        is_training = False
        batch_size = cfg.training.batch_size

        val_idxs = VAL_INDICES
        num_batches = int(np.ceil(len(val_idxs) / batch_size))
        num_full_batches = int(np.floor(len(val_idxs) / batch_size))

        loss_sum = 0
        
        #  step_in_epochs = epoch + 1
        eval_dir = f'{cfg.logging.logdir}/val/eval{str(epoch).zfill(6)}'
        base_eval_dir = eval_dir

        if os.path.isdir(eval_dir):
            os.rename(eval_dir, f'{eval_dir}_backup_{int(time.time())}')
        os.makedirs(eval_dir, exist_ok=True)

        # Prediction containers
        all_pred_translations = np.empty((len(val_idxs), 3), dtype=np.float32)
        all_pred_angles = np.empty((len(val_idxs), 1), dtype=np.float32)
        
        # The conversion from logits to is done outside the model
        # Pass this to the model!
        all_pred_s2_pc1angles = np.empty((len(val_idxs), 1), dtype=np.float32)
        all_pred_s2_pc2angles = np.empty((len(val_idxs), 1), dtype=np.float32)
        
        # Ground truth contina
        all_gt_translations = np.empty((len(val_idxs), 3), dtype=np.float32)
        all_gt_angles = np.empty((len(val_idxs), 1), dtype=np.float32)
        all_gt_pc1centers = np.empty((len(val_idxs), 3), dtype=np.float32)
        
        cumulated_times = 0.
        for batch_idx in range(num_batches):
            
            # AQUI ERROR!
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(val_idxs))
            
            print(f'----- Samples {start_idx}/{len(VAL_INDICES)} -----')
            
            # TODO: Create a class to solve this mess
            pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles = provider.load_batch(val_idxs[start_idx:end_idx], path=ppath)
            #print(pc1centers)
            
            # TODO: Investigate a better way to do this feed_dict
            feed_dict = {
                ops['pcs1']: pcs1,
                ops['pcs2']: pcs2,
                ops['translations']: translations,
                ops['rel_angles']: rel_angles,
                ops['is_training_pl']: is_training,
                ops['pc1centers']: pc1centers,
                ops['pc2centers']: pc2centers,
                ops['pc1angles']: pc1angles,
                ops['pc2angles']: pc2angles,
            }
            start = time.time()
            
            # TODO: IDEM Create class to solve this mess
            summary, loss_val, pred_translations,pred_pc1angle_logits, pred_pc2angle_logits, pred_remaining_angle_logits, _, _, _, _ = sess.run([ops['merged'], ops['loss'], ops['pred_translations'], ops['pred_pc1angle_logits'], ops['pred_pc2angle_logits'], ops['pred_remaining_angle_logits'], ops['pred_s1_pc1centers'], ops['pred_s1_pc2centers'], ops['pred_s2_pc1centers'], ops['pred_s2_pc2centers']], feed_dict=feed_dict)
            
            # Why do we need time?
            cumulated_times += time.time() - start
            # ?
            actual_batch_size = end_idx - start_idx
            
            # How can this be longer? Maybe when not full batch
            pred_translations = pred_translations[:actual_batch_size]
            # Correct from logits to angle
            pred_angles_pc1 = MODEL.classLogits2angle(pred_pc1angle_logits[:actual_batch_size])
            pred_angles_pc2 = MODEL.classLogits2angle(pred_pc2angle_logits[:actual_batch_size])
            pred_angles_remaining = MODEL.classLogits2angle(pred_remaining_angle_logits[:actual_batch_size])
            # Final angle computation
            pred_angles = pred_angles_pc2 - pred_angles_pc1 + pred_angles_remaining
            
            # Why this?
            if actual_batch_size == batch_size:  # last batch is not counted
                loss_sum += loss_val
            
            # Some parameters (ARE THEY NEEDED?)
            mean_per_transform_loss = loss_sum / num_full_batches if num_full_batches > 0 else 0.
            mean_execution_time = cumulated_times / float(len(val_idxs))
            
            print(f"{loss_val}")
            
            # Store result to big array TODO: CONVERT TO SINGLE LINE OP
            for idx in range(actual_batch_size):
                global_idx = start_idx + idx

                all_pred_translations[global_idx] = pred_translations[idx]
                all_pred_angles[global_idx] = pred_angles[idx]

                all_gt_translations[global_idx] = translations[idx]
                all_gt_angles[global_idx] = rel_angles[idx]
                all_gt_pc1centers[global_idx] = pc1centers[idx]

    import matplotlib.pyplot as plt
    plt.scatter(all_gt_pc1centers[:,0], all_gt_pc1centers[:,1])
    plt.show()

    print("Results fully computed")

    info = np.hstack((all_pred_translations[:,:-1], all_pred_angles, all_gt_translations[:,:-1], all_gt_angles, all_gt_pc1centers[:,:-1]))
    names = ['pred_trans_x', 'pred_trans_y', 'pred_angles', 'gt_trans_x', 'gt_trans_y', 'gt_angles', 'gt_pc1centers_x', 'gt_pc1centers_y']

    df = DataFrame(info, columns=names)
    
    return df

def read_paths(filepath):
    """ Read path json file from filepath """
    with open(filepath) as file:
        name_cont = json.load(file)
    return name_cont


if __name__ == "__main__":
    # Path definition
    home_path = '/home/usuario/'    # Adjust your path!
    datasets_path = os.path.join(home_path, 'project_data', 'datasets')
    new_path = os.path.join(home_path, 'project_data', 'new_datasets') 
    
    dataset_name = 'KITTITrackletsCarsPersons'
    
    dataset_path = os.path.join(datasets_path, dataset_name)

    json_path = os.path.join(new_path, dataset_name,f"{dataset_name}_path.json")
    
    # Old way of getting trajectories
    #all_trajectories = read_paths(json_path)
    #all_ids = [val for val in all_trajectories.values()]
    #all_ids = [idx for list_ in all_ids for idx in list_]

    df = read_csv(f'/home/usuario/project_data/new_datasets/{dataset_name}/{dataset_name}_eval_info.csv')

    points = []
    for entry in df.values:
        points.extend(range(entry[2],entry[2]+entry[3]))
    all_ids = [str(i).zfill(8) for i in points]

    config=f'configs/{dataset_name}.json'
    eval_epoch='4' 
    refineICP=False
    refineICPmethod='p2p'

    configs = [config, eval_epoch, refineICP, refineICPmethod]

    results = run_inference(configs, all_ids, ppath=dataset_path)

    results.to_csv(os.path.join(new_path, dataset_name, f'{dataset_name}_output.csv'), index= False)   
    
    print("Results stored to CSV file")


