# MEAN AND COVARIANCE SCRIPT
# This scripts computed the mean and covariance for each observation
# (pointcloud) of the dataset. It is used in the Kalman Filter.

import open3d as o3d
import numpy as np
import os

def get_mean_cov(pointcloud_path):
    ''' Returns the mean and covariance of pcd
    '''
    arr = np.load(pointcloud_path) # size: nx3

    # Create poincloud and load points in it
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
   

    # Return the mean and covariance
    return o3d.compute_point_cloud_mean_and_covariance(pcd)


if __name__ == "__main__":

    # Path and file
    home_path = '/home/usuario/'    # Adjust your path!
    datasets_path = os.path.join(home_path, 'project_data', 'datasets')

    # This are all the KITTI datasets we can use
    KITTIDatasets = ['KITTITrackletsCars', 'KITTITrackletsCarsPersons', 'KITTITrackletsCarsHard', 'KITTITrackletsCarsPersonsHard']

    for name in KITTIDatasets:
        for obs in [1,2]:
            dataset_path = os.path.join(datasets_path, name)
            export_path = os.path.join(dataset_path, f'pointcloud{obs}')
            file_list = sorted(os.listdir(export_path))
            
            if os.path.exists(dataset_path+f'/observed_mean_pc{obs}.npy'):
                print(dataset_path+f'/observed_mean_pc{obs}.npy', "already exported!")
                continue

            # Data containers
            means = np.zeros((len(file_list),3))
            covs = np.zeros((len(file_list),3,3))

            for i,filename in enumerate(file_list):
                # Compute info
                pcd_path = os.path.join(export_path, filename)
                info = get_mean_cov(pcd_path)

                means[i] = info[0]
                covs[i] = info[1]

                if i % 500 == 0: print(f'Processing {i}th pointcloud')
            
            
            np.save(dataset_path+f'/observed_mean_pc{obs}', means)
            np.save(dataset_path+f'/observed_covs_pc{obs}', covs)

            print(name, "exported!")
    
    
    


