# create_new_dataset.py Jaume Colom - jaume.colom@tum.de
# This script converts the raw_dataset(original align3d datasets)
# to the format required for the actual project. It creates the necessary  
# folder structure and files for object tracking.

import numpy as np
import json
import os

# Utils to fix the raw metas into usable ones
def to_arr(value):
    """ Converts a string with float values to an actual list of floats """
    return [float(name) for name in value.split()]

def fix_meta(meta):
    """ Converts all the string coded values to lists """
    meta['start_position'] = to_arr(meta['start_position'])
    meta['end_position'] = to_arr(meta['end_position'])
    meta['translation'] = to_arr(meta['translation'])


import pathlib
def mk_folder(path):    
    """ Util to create folders NO COMPLAINS! """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# Functions to store single seq_ids and track_ids to folders
def store_single_path(curr_folder_path, curr_trajectory):
    """ Stores a single path """
    for time_point in curr_trajectory:
        outputfile_path = os.path.join(curr_folder_path, f"{time_point['filename']}.json")
        with open(outputfile_path, 'w') as json_file:
            json.dump(time_point, json_file, indent=4)
    
def store_seq(new_path, seq_track_metas):
    """ Stores a full sequence """
    for seq_id in seq_track_metas.keys():
        print(f"\tConverting {seq_id}...")
        for track_id in seq_track_metas[seq_id].keys():
            # Create path
            curr_folder_path = os.path.join(new_path,str(seq_id),str(track_id))
            mk_folder(curr_folder_path)
            # Get individual paths
            curr_trajectory = seq_track_metas[seq_id][track_id]
            # Store them
            store_single_path(curr_folder_path, curr_trajectory)


# Main functions
def load_all_metas(dataset_path):
    """ Loads all jsons and stores them in a list(container) """
    container = list()
    meta_path = os.path.join(dataset_path, "meta")
    for filename in sorted(os.listdir(meta_path)):
        # Create path and load file
        file_path = os.path.join(meta_path, filename)
        with open(file_path) as json_file: meta_dict = json.load(json_file)
        # Convert the string to lists
        fix_meta(meta_dict)
        meta_dict['filename'] = filename[:-5]
        # Append to file
        container.append(meta_dict)
    
    return container

def slice_seq_track(all_metas): 
    """ Slices a full dataset by 'seq' and 'track_id' 
        Reads through each meta file and filters these by the indicated parameters.
        Parameters:
        - all_metas: Dict containing all the meta jsons (in dict format)
        Returns:
        - container: Nested dicts containing the different meta files; in the first level
        the key is the 'seq_id' and the value is another dict; the second dict contains the
        'track_id' as key and a list with all the ordered metas as value.
    """
    # Get unique seq_ids in the list of dicts
    seqs = list(set(meta['seq'] for meta in all_metas)) 

    # Store the correct meta files in each dict key (seq_ids)
    seq_metas = {seq: [] for seq in seqs}
    for meta in all_metas: 
        seq_metas[meta['seq']].append(meta)

    seq_track_metas = dict()

    for seq_id in seq_metas.keys():    
        # This is a list containing all the meta dicts of specific seq_id
        curr_metas = seq_metas[seq_id] 

        # Get unique trackids in the dicts
        track_ids = list(set(meta['trackids'][0] for meta in curr_metas)) 

        # Store the different metas at each key (track_ids)
        track_metas = {idn: [] for idn in track_ids}
        for meta in curr_metas: 
            track_metas[meta['trackids'][0]].append(meta)

        # The seq_id is used as key to store the computed dict
        # The output is a dict of dicts
        seq_track_metas[seq_id] = track_metas
    
    return seq_track_metas

def create_name_container(seq_track_metas):
    """ Creates a file containing all the number_id observations for each path 
        It serves as ground thruth for later training and evaluating the model, 
        as we need paths for object tracking.
        Params:
        - seq_track_metas: the output from slice_seq_track()
        Returns:
        - name_container: dict with the format (string)'seq_id':(list)'meta_files'(as dict)
    """
    name_container = dict()
    # Iterate through every value in the dataset
    for seq_id in seq_track_metas.keys():
        for track_id in seq_track_metas[seq_id].keys():
            # This line gets each filename in the current path
            names = [meta['filename'] for meta in seq_track_metas[seq_id][track_id]]
            # 
            name_container[f'{seq_id}_{track_id}'] = names
            
    return name_container

def dump_name_container(name_cont, filepath):
    """ Exports name_cont(dict) to filepath in json format """
    with open(filepath, 'w') as outfile:
        json.dump(name_cont, outfile)
    print(f"Exported to {filepath}")

if __name__ == "__main__":
    # Path definition
    home_path = '/home/usuario/'    # Adjust your path!
    datasets_path = os.path.join(home_path, 'project_data', 'datasets')
    new_path = os.path.join(home_path, 'project_data', 'new_datasets')

    # This are all the KITTI datasets we can use
    KITTIDatasets = ['KITTITrackletsCars', 'KITTITrackletsCarsPersons', 'KITTITrackletsCarsHard', 'KITTITrackletsCarsPersonsHard']

    for name in KITTIDatasets:
        print(f"Now converting {name}")
        
        # 1.Load all metas from dataset
        # datasets_path should be from class
        dataset_path = os.path.join(datasets_path, name)
        all_metas = load_all_metas(dataset_path)
        
        # 2.Slice datasets by seq_id and track_id
        seq_track_metas = slice_seq_track(all_metas)
        
        # 3.Store paths
        # Create paths and store
        destination_path = os.path.join(new_path, f'{name}_path.json')
        name_container = create_name_container(seq_track_metas)
        dump_name_container(name_container, destination_path)
        # mk_folder(new_path)
        # destination_path = os.path.join(new_path, name)
        # store_seq(destination_path, seq_track_metas)

    print("Done!")