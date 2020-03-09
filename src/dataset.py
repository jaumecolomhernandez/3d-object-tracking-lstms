import pandas as pd
import numpy as np

import os
import sys

from kalmanfiltermotion import KalmanMotionTracker

class Route:
    """ Class to contain all the route data 
        Params:
        - name: (string) Name of the track; Used to identify tracks in the list.
        - full: (bool) Flag to indicate if this route has got full measurements information
        or it only has metainfo.
        - generated: (DataFrame) Contains info generated from nn, kf or a combination of both.
        - ground_truth: (DataFrame) Contains ground truth (from metafile) for each timestamp.
        - routes: (DataFrame) Contains the position (x,y,a) per timepoint. Used for plotting or
        computing error.
        - error: (DataFrame) Contains the error computation per timepoint. It is agregated to compute
        the full track error.
        TBC.
    """
    
    def __init__(self, name, base_data):
        # Basic parameters
        self.name = name
        self.full = True
        self.n_samples = base_data.shape[0]

        # Error parameters
        self.trans_error = dict()
        self.angle_error = dict()
        
        # Support parameters
        n_timestamps = base_data.shape[0]
        df_index = np.arange(n_timestamps)
        
        # Remove the index for the dataframe
        base_data.reset_index(drop=True, inplace=True)
        
        # Create dataframe for each table
        # Generated information -> align3d, kf, lstm
        self.generated = pd.DataFrame(0, index=df_index, columns=['align3d_x', 'align3d_y', 'align3d_a'])
        self.generated[['align3d_x', 'align3d_y', 'align3d_a']] = base_data[['pred_trans_x', 'pred_trans_y','pred_angles']]
        
        # Info from the meta files -> translation and pc1centers
        self.ground_truth = pd.DataFrame(0, index=df_index, columns=['x', 'y', 'a', 'a1_x', 'a1_y', 'a1_a'])
        self.ground_truth[['x', 'y', 'a', 'a1_x', 'a1_y', 'a1_a']] = base_data[['gt_trans_x', 'gt_trans_y', 'gt_angles', 'gt_pc1centers_x', 'gt_pc1centers_y', 'gt_pc1angles']]
        
        # Generated routes ex. -> pc1center+gt_trans, cumulative routes based on preds
        self.routes = pd.DataFrame(0, index=df_index, columns=['gt_x', 'gt_y', 'gt_a'])
        # Compute the ground truth route
        pos = [self.ground_truth['a1_x']+self.ground_truth['x'], \
                      self.ground_truth['a1_y']+self.ground_truth['y']]
        angle = self.ground_truth['a1_a']+self.ground_truth['a']

        self.routes.loc[:,'gt_x'] = pos[0]
        self.routes.loc[:,'gt_y'] = pos[1]
        self.routes.loc[:,'gt_a'] = angle
        
        # Computed errors for each prediciton
        self.error = pd.DataFrame(0, index=df_index, columns=[])
    
    def __str__(self):
        pass
        
    def make_relative_route(self, cat):
        """ Creates relative route
            A relative route is pc1centers + translation prediction. It does not take
            into account the agregated error.
        """
        # Compute pose
        fake_pos = [self.ground_truth['a1_x']+self.generated[f'{cat}_x'], \
                      self.ground_truth['a1_y']+self.generated[f'{cat}_y']]
        fake_angle = self.ground_truth['a1_a']+self.generated[f'{cat}_a']

        # Store to table
        self.routes.loc[:,f'{cat}_relative_x'] = fake_pos[0]
        self.routes.loc[:,f'{cat}_relative_y'] = fake_pos[1]
        self.routes.loc[:,f'{cat}_relative_a'] = fake_angle

    def make_absolute_route(self, cat):
        """ Computes route
            It starts from the first pc1 center and keeps adding the translation predictions.
            Parameters:
            cat: (string) 
        """
        # Data container
        route_arr = np.zeros((self.ground_truth.shape[0], 3))
        # This is the start position (pc1center)
        curr_pos = np.array([self.ground_truth.loc[0,'a1_x'], self.ground_truth.loc[0,'a1_y']])
        curr_angle = self.ground_truth.loc[0,'a1_a']

        # Iterate 
        for i_r in range(self.n_samples):
            # Compute timepoint position and angle
            curr_pos = curr_pos[0]+self.generated.loc[i_r, f'{cat}_x'], curr_pos[1]+self.generated.loc[i_r, f'{cat}_y']
            curr_angle = curr_angle+self.generated.loc[i_r, f'{cat}_a']
            # Store angle in container
            route_arr[i_r,:2] = curr_pos
            route_arr[i_r,2] = curr_angle

        self.routes.loc[:,f'{cat}_absolute_x'] = route_arr[:,0]
        self.routes.loc[:,f'{cat}_absolute_y'] = route_arr[:,1]
        self.routes.loc[:,f'{cat}_absolute_a'] = route_arr[:,2]
    
    def run_kalman_filter(self, cat):
        """  """
        # Tracker object initialization
        position = self.generated.iloc[0,0:3].values
        tracker = KalmanMotionTracker(position, None)
        
        # Container for the KF data
        kf = np.zeros((self.n_samples,3))
        kf[0,:] = position

        # We feed from 0 to N observations to the filter
        for i in range(1, self.n_samples):
            position = self.generated.loc[i,[f'{cat}_x', f'{cat}_y', f'{cat}_a']].values
            tracker.update(position)
            # We store the inmediate pose
            predictions = tracker.predict()
            # We present the updated state
            predictions = tracker.get_state()
            kf[i,:] = predictions
        
        # Store them in the container
        self.generated.loc[:,f'{cat}+kf_x'] = kf[:,0]
        self.generated.loc[:,f'{cat}+kf_y'] = kf[:,1]
        self.generated.loc[:,f'{cat}+kf_a'] = kf[:,2]

    def compute_rmse_error(self, cat):    
        """ Computes the rmse for translation and angle.
            Params:
            cat: (string) Contains the metric on wich to compute the error
        """
        # squared
        self.error.loc[:,f'{cat}_x'] = (self.routes[f'{cat}_x']-self.routes['gt_x'])**2
        self.error.loc[:,f'{cat}_y'] = (self.routes[f'{cat}_y']-self.routes['gt_y'])**2
        self.error.loc[:,f'{cat}_a'] = ((self.routes[f'{cat}_a']-self.routes['gt_a'])*180)**2

        # mean
        x_e = np.mean(self.error[f'{cat}_x'])
        y_e = np.mean(self.error[f'{cat}_y'])
        a_e = np.mean(self.error[f'{cat}_a'])
        # root
        trans = (x_e+y_e)**0.5
        angle = a_e**0.5

        # TODO: Add log
        #print(f"RMSE Trans: {trans}\nRMSE Angle: {angle}")

        self.trans_error[cat] = trans
        self.angle_error[cat] = angle

        return trans, angle
    
    
    def plot_route(self):
        pass