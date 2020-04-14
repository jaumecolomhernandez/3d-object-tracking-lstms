# Adapted from: https://github.com/xinshuoweng/AB3DMOT/blob/master/main.py
# Credits: Xinshuo Weng (https://github.com/xinshuoweng)

from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanMotionTrackerMeansAlignXY(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  def __init__(self, position, parameter, route):
    """
    Initialises a tracker using initial position.

    KF Instance variables:
      x : ndarray (dim_x, 1), default = [0,0,0…0] filter state estimate
      P : ndarray (dim_x, dim_x), default eye(dim_x) covariance matrix
      Q : ndarray (dim_x, dim_x), default eye(dim_x) Process uncertainty/noise
      R : ndarray (dim_z, dim_z), default eye(dim_x) measurement uncertainty/noise
      H : ndarray (dim_z, dim_x) measurement function
      F : ndarray (dim_x, dim_x) state transistion matrix
      B : ndarray (dim_x, dim_u), default 0 control transition matrix
    """
    # Constant to check if 
    self.route = route
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=4)   

    # x = [x,y,vx,vy,va]
    self.kf.x = position.reshape((4, 1))
    #self.kf.P[3:,3:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    self.kf.P = np.eye(4) * 50.
    #self.kf.R = np.eye(5) * 0.5
    parameter = 0
    self.kf.R = np.array([[parameter,0,0,0],      # Measurement uncertainty
                          [0,parameter,0,0],
                          [0,0,500,0],
                          [0,0,0,500]])
    #self.kf.R = np.eye(3) * parameter


    self.kf.P *= 10
    self.kf.Q = np.eye(4) * 100

    self.kf.F = np.array([[1,0,1,0],      # state transition matrix
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])
    
    self.kf.H = np.array([[1,0,0,0],      # measurement function,
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])
 

    # self.kf.R[0:,0:] *= 10.   # measurement uncertainty

    self.time_since_update = 0
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0

  def update(self, position): 
    """ 
    Updates the state vector with observed position.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    # ######################### orientation correction NEEDED?
    # if self.kf.x[2] >= np.pi: self.kf.x[2] -= np.pi * 2    # make the theta still in the range
    # if self.kf.x[2] < -np.pi: self.kf.x[2] += np.pi * 2

    # new_theta = position[2]
    # if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    # if new_theta < -np.pi: new_theta += np.pi * 2
    # position[2] = new_theta

    # predicted_theta = self.kf.x[2]
    # if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
    #   self.kf.x[2] += np.pi       
    #   if self.kf.x[2] > np.pi: self.kf.x[2] -= np.pi * 2    # make the theta still in the range
    #   if self.kf.x[2] < -np.pi: self.kf.x[2] += np.pi * 2
      
    # # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    # if abs(new_theta - self.kf.x[2]) >= np.pi * 3 / 2.0:
    #   if new_theta > 0: self.kf.x[2] += np.pi * 2
    #   else: self.kf.x[2] -= np.pi * 2
    
    # #########################
    self.kf.update(position)

    # if self.kf.x[2] >= np.pi: self.kf.x[2] -= np.pi * 2    # make the theta still in the range
    # if self.kf.x[2] < -np.pi: self.kf.x[2] += np.pi * 2

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      

    # if self.kf.x[2] >= np.pi: self.kf.x[2] -= np.pi * 2
    # if self.kf.x[2] < -np.pi: self.kf.x[2] += np.pi * 2
    
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False

    self.time_since_update += 1
    self.history.append(self.kf.x)

    if self.route:
      return self.kf.x[:2].reshape((2, ))
    else:
      return self.kf.x[2:].reshape((2, ))

  def get_state(self):
    """
    Returns the current motion estimate.
    """

    if self.route:
      return self.kf.x[:2].reshape((2, ))
    else:
      return self.kf.x[2:].reshape((2, ))