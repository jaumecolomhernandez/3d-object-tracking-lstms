from dataset import Route
import numpy as np
import os, sys
import pandas as pd

def routes_to_dict(eval_info, predictions, mean1, mean2, median1, median2):
    ''' Loads (meta)info from disk to memory classes for easy manipulation '''
    routes = dict()
    n_routes = eval_info.shape[0]
    
    for i in range(n_routes):
        current = eval_info.iloc[i]
        start_i = current['cumsum']
        n_points = current['n_points']
        current_pred = predictions.iloc[start_i:start_i+n_points]
        
        all_i = current['first_point']
        sliced_mean1 = mean1[all_i:all_i+n_points,:]
        sliced_mean2 = mean2[all_i:all_i+n_points,:]
        sliced_median1 = median1[all_i:all_i+n_points,:]
        sliced_median2 = median2[all_i:all_i+n_points,:]
        sliced_pred1 = predictions.loc[start_i:start_i+n_points-1,['pred_pc1center_x', 'pred_pc1center_y']].values
        sliced_pred2 = predictions.loc[start_i:start_i+n_points-1,['pred_pc2center_x', 'pred_pc2center_y']].values
        
        route_obj = Route(current['name'], current_pred)
        
        # Add the observation to the gt table
        route_obj.add_observation(obs_pc1=sliced_mean1, obs_pc2=sliced_mean2, name="mean")
        route_obj.add_observation(obs_pc1=sliced_median1, obs_pc2=sliced_median2, name="median")
        route_obj.add_observation(obs_pc1=sliced_pred1, obs_pc2=sliced_pred2, name="pred")

        routes[current['name']] = route_obj
            
    return routes

def compute_kf_param(routes_dict, param):
    for key, route in routes_dict.items():
        # nn predictions come from neural network! (input data to this script)
        # make align3d+kf predictions
        # route.run_kalman_filter_means(store_name='kf/means', generates_route=True, param=param)
        route.run_kalman_filter_means_align_xy(store_name='kf/align3d+pred_xy', generates_route=False, param=param)

        # Computes all the routes
        route.compute_routes()
        # Computes all the route errors
        route.compute_all_rmse()
        #route.compute_rmse_error('kf/align3d+means')
        
    # Compute error
    error = 0
    for name, route in routes_dict.items():
        error += route.trans_error['kf/align3d+pred_xy_absolute']
    actual_error = error/len(routes_dict)
    
    
    # Print info
    print(f"With param: {param} error: {actual_error:.4f}")
    
    return actual_error

if __name__ == "__main__":

    home_path = '/home/usuario/'

    datasets_path = os.path.join(home_path, 'project_data', 'datasets')

    all_datasets = ['KITTITrackletsCars', 'KITTITrackletsCarsPersons', 'KITTITrackletsCarsHard', 'KITTITrackletsCarsPersonsHard']

    dataset_path = os.path.join(datasets_path, all_datasets[0])

    # This is the output from the NN
    predictions = pd.read_csv(os.path.join(dataset_path, "NN_output.csv"))
    # This gives information about the paths
    eval_info = pd.read_csv(os.path.join(dataset_path, "info_eval.csv"))
    # Observation means for pc1 and pc2
    means1 = np.load(os.path.join(dataset_path, 'observed_mean_pc1.npy'))
    means2 = np.load(os.path.join(dataset_path, 'observed_mean_pc2.npy'))
    # Observation medians for pc1 and pc2
    medians1 = np.load(os.path.join(dataset_path, 'observed_median_pc1.npy'))
    medians2 = np.load(os.path.join(dataset_path, 'observed_median_pc2.npy'))

    # Angles correction Reduces the MSE from 200 to 66 degrees.
    predictions.loc[predictions['pred_angles']>3, 'pred_angles'] = predictions[predictions['pred_angles']>3]['pred_angles'] - 3
    predictions.loc[predictions['pred_angles']<-3, 'pred_angles'] = predictions[predictions['pred_angles']<-3]['pred_angles'] + 3

    print("All data ready - everything read!")
    routes_dict = routes_to_dict(eval_info, predictions, mean1=means1, mean2=means2, median1=medians1, median2=medians2)

    print("Starting computation - routes_dict ready")
    for i in np.arange(0, 10, 1):
        compute_kf_param(routes_dict, i)
