# MEDIAN SCRIPT
# This scripts computed the mean and covariance for each observation
# (pointcloud) of the dataset. It is used in the Kalman Filter.

import numpy as np
import os

def get_median(pointcloud_path):
    ''' Returns the median of a pointcloud
    '''
    arr = np.load(pointcloud_path) # size: nx3
    
    # Return the mean and covariance
    return np.median(arr,axis=0)[:3]


if __name__ == "__main__":

    # Path and file
    home_path = '/home/usuario/'    # Adjust your path!
    datasets_path = os.path.join(home_path, 'project_data', 'datasets')

    # This are all the KITTI datasets we can use
    KITTIDatasets = ['KITTITrackletsCars', 'KITTITrackletsPersons', 'KITTITrackletsCarsPersons', 'KITTITrackletsCarsHard', 'KITTITrackletsCarsPersonsHard']

    for name in KITTIDatasets:
        for obs in [1,2]:
            dataset_path = os.path.join(datasets_path, name)
            export_path = os.path.join(dataset_path, f'pointcloud{obs}')
            file_list = sorted(os.listdir(export_path))
            
            if os.path.exists(dataset_path+f'/observed_median_pc{obs}.npy'):
                print(dataset_path+f'/observed_median_pc{obs}.npy', "already exported!")
                continue

            # Data containers
            medians = np.zeros((len(file_list),3))

            for i,filename in enumerate(file_list):
                # Compute info
                pcd_path = os.path.join(export_path, filename)
                info = get_median(pcd_path)

                medians[i] = info

                if i % 500 == 0: print(f'Processing {i}th pointcloud')
            
            
            np.save(dataset_path+f'/observed_median_pc{obs}', medians)

            print(name, "exported!")
    
    
    


