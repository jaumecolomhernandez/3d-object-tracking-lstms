from dataset import Route
import numpy as np
import os, sys
import pandas as pd

def routes_to_dict(eval_info, predictions, means1, means2):
    ''' Loads (meta)info from disk to memory classes for easy manipulation '''
    routes = dict()
    n_routes = eval_info.shape[0]
    
    for i in range(n_routes):
        current = eval_info.iloc[i]
        start_i = current['cumsum']
        n_points = current['n_points']
        current_pred = predictions.iloc[start_i:start_i+n_points]
        
        all_i = current['first_point']
        route_means1 = means1[all_i:all_i+n_points,:]
        route_means2 = means2[all_i:all_i+n_points,:]
        
        route_obj = Route(current['name'], current_pred)
        
        # Add the observation to the gt table
        route_obj.add_obs_mean(route_means1, route_means2)
        
        routes[current['name']] = route_obj
            
    return routes

def compute_kf_param(routes_dict, param):
    for key, route in routes_dict.items():
        # nn predictions come from neural network! (input data to this script)
        # make align3d+kf predictions
        route.run_kalman_filter(cat='align3d', store_name='+kf/obs', use_obs=True, param=param)
        route.make_absolute_route('align3d+kf/obs')

        # compute kf/obs error
        route.compute_rmse_error('align3d+kf/obs_absolute')
        
    # Compute error
    error = 0
    for name, route in routes_dict.items():
        error += route.trans_error['align3d+kf/obs_absolute']
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

    # Angles correction Reduces the MSE from 200 to 66 degrees.
    predictions.loc[predictions['pred_angles']>3, 'pred_angles'] = predictions[predictions['pred_angles']>3]['pred_angles'] - 3
    predictions.loc[predictions['pred_angles']<-3, 'pred_angles'] = predictions[predictions['pred_angles']<-3]['pred_angles'] + 3

    print("All data ready - everything read!")
    routes_dict = routes_to_dict(eval_info, predictions, means1, means2)

    print("Starting computation - routes_dict ready")
    for i in np.arange(0.5,5,0.5):
        compute_kf_param(routes_dict, i)