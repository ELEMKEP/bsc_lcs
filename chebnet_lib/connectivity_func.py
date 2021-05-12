# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:38:47 2017

@author: JANG-LAB
"""

import numpy as np
import pickle
import scipy.spatial
import scipy.sparse
import scipy.signal
import itertools
from sklearn.metrics import mutual_info_score
import multiprocessing

N_POOLS = 10

def conn_xcorr(data):
  """Calculate the connectivity to construct graph adjacency matrix by cross-correlation.

  Keyword arguments:
      data -- Original data
      dimension: n_trial x channel_dim x timestamps

  Return arguments:
      G -- Calculated connectivity
      
      
  """

  n_trial = data.shape[0]
  channel_dim = data.shape[1]
  G = np.zeros([n_trial, channel_dim, channel_dim])

  for trial_idx in range(n_trial):
    trial_data = data[trial_idx]
    col_g, row_g = np.meshgrid(range(channel_dim), range(channel_dim))
    for (idx1, idx2) in zip(row_g.reshape(-1), col_g.reshape(-1)):
      xcorr = np.correlate(trial_data[idx1], trial_data[idx2], 'full')
      G[trial_idx][idx1][idx2] = np.max(np.abs(xcorr))

  G = (G + np.transpose(G, (0, 2, 1)))/2 # to guarantee symmetricity
  return G
  

def graph_max_k(data, sort='min', k=0):
  '''
  return k maximum/minimum elements from the graph
  '''
  data = data.copy()

  assert (data.shape[0] == data.shape[1]), 'input matrix should be a square matrix.'
  N = data.shape[0]

  if sort == 'min':
    if k != 0:
      d = np.argsort(data, axis=1)[:, k:]
      ir = np.mgrid[0:N, k:N][0]
      data[ir, d] = 0
  elif sort == 'max':
    if k != 0:
      d = np.argsort(data, axis=1)[:, :-k]
      ir = np.mgrid[0:N, 0:(N-k)][0]
      data[ir, d] = 0
  else:
    print('sort method should be one of min or max.')
    return None

  data = (data + np.transpose(data)) / 2
  data[np.diag_indices(N)] = 0
    
  return data
  
  
def conn_xcorr_fisherz(data, k=0):
  """Calculate the connectivity to construct graph adjacency matrix by 
  average cross-correlation by fisher's Z.

  Keyword arguments:
      data -- Original data
          dimension: n_trial x channel_dim x timestamps
      k -- number of highest correlations that remains in the result matrix.

  Return arguments:
      nR -- averaged cross-correlation matrix based on fisher's Z
  
  """
  n_trial = data.shape[0]
  channel_dim = data.shape[1]
  R = np.zeros([n_trial, channel_dim, channel_dim])

  pool = multiprocessing.Pool(N_POOLS)
  R = np.array(pool.map(np.corrcoef, [data[trial_idx] for trial_idx in range(n_trial)]))
  # for trial_idx in range(n_trial):
  #   R[trial_idx] = np.corrcoef(data[trial_idx])

  Z = np.arctanh(R)
  nZ = np.mean(Z, axis=0)
  
  nR = np.tanh(nZ)
  
  if k != 0:
    d = np.argsort(nR, axis=1)[:, :-k]
    ir = np.mgrid[0:channel_dim, 0:(channel_dim-k)][0]
    nR[ir, d] = 0
    
  nR = (nR + np.transpose(nR))/2 # to guarantee symmetricity
  
  nR[np.diag_indices(channel_dim)] = 0
  
  return nR
    
def sph_dist(u, v):
  th = np.deg2rad(np.abs(u[0] - v[0]))
  ph1 = np.deg2rad(u[1])
  ph2 = np.deg2rad(v[1])

  s = np.arccos(
    np.sin(ph1) * np.sin(ph2) + np.cos(ph1) * np.cos(ph2) * np.cos(th))
  d = u[2] * s
  return d

def get_distance_matrix(dist_metric='euclidean'):
  # Only 3d cartesian distance
  twente_to_geneva = [0, 1, 3, 2, 5, 4, 7, 6, \
                      9, 8, 11, 10, 13, 14, 15, 12, \
                      29, 28, 30, 26, 27, 24, 25, 31, \
                      22, 23, 20, 21, 18, 19, 17, 16]

  tw_labels = list()
  tw_2d_sph_coords = list()
  tw_3d_car_coords = list()
  tw_3d_sph_coords = list()
  with open('10-20_32ch.ced', 'rb') as f:
    f.readline()
    while True:
      a = f.readline()

      asp = str.split(str(a, 'utf-8'))
      if len(asp) < 10:
        break
      tw_labels.append(asp[1])
      tw_2d_sph_coords.append(np.asarray(asp[2:4]).astype(np.float32))
      tw_3d_car_coords.append(np.asarray(asp[4:7]).astype(np.float32))
      tw_3d_sph_coords.append(np.asarray(asp[7:10]).astype(np.float32))

  ge_labels = list()
  ge_2d_sph_coords = list()
  ge_3d_car_coords = list()
  ge_3d_sph_coords = list()
  for idx in twente_to_geneva:
    ge_labels.append(tw_labels[idx])
    ge_2d_sph_coords.append(tw_2d_sph_coords[idx])
    ge_3d_car_coords.append(tw_3d_car_coords[idx])
    ge_3d_sph_coords.append(tw_3d_sph_coords[idx])

  ge_2d_sph_coords = np.asarray(ge_2d_sph_coords)
  ge_3d_car_coords = np.asarray(ge_3d_car_coords)
  ge_3d_sph_coords = np.asarray(ge_3d_sph_coords)
  
  d = scipy.spatial.distance.pdist(ge_3d_car_coords, metric=dist_metric)
  d = scipy.spatial.distance.squareform(d)
  
  return d
  
def mutual_information(data, bins=None, axis=-2, fs=128):
  '''
  Calculate mutual information between channel pairs
  '''
  nv = data.shape[axis]
  data_axswap = np.moveaxis(data, axis, 0)
  data_axswap = np.reshape(data_axswap, (nv, -1))
  
  # np.histogram bins
  # [1,2,3,4] --> [1,2), [2,3), [3,4]
  if bins is None:
    bins = np.percentile(data, np.arange(0, 101))
  
  mi_mat = np.zeros((nv, nv))
  # mutual information
  for i in range(nv-1):
    print(i)
    for j in range(i+1, nv):
      # cx == np.sum(c_xy, axis=1)
      # cy == np.sum(c_xy, axis=0)
      c_xy, _, _ = np.histogram2d(data_axswap[i], data_axswap[j], bins)
      c_x, _ = np.histogram(data_axswap[i], bins)
      c_y, _ = np.histogram(data_axswap[j], bins)
      
      c_xy = c_xy / np.sum(c_xy)
      c_x = c_x / np.sum(c_x)
      c_y = c_y / np.sum(c_y)
      
      mi = 0
      for k in range(len(bins)-1):
        for l in range(len(bins)-1):
          if c_xy[k, l] != 0:
            logf = np.log(c_xy[k, l]/(c_x[k]*c_y[l]))
          else:
            logf = 0
          mi = mi + c_xy[k,l] * logf
      mi_mat[i, j] = mi
      mi_mat[j, i] = mi
      
  return mi_mat
  
  
def magnitude_squared_coherence(data, fs, window, noverlap, band=[4, 45], axis=-2):
  nv = data.shape[axis]

  data_axswap = np.moveaxis(data, axis, 0)
  data_axswap = np.reshape(data_axswap, (nv, -1, fs*60))

  n_samples = data_axswap.shape[1]
  fn = np.int(len(window)/2)

  msc_mat = np.zeros((nv, nv, fn+1))
  print(msc_mat.shape)

  for i in range(nv-1):
    print(i)
    for j in range(i+1, nv):
      for k in range(n_samples):
        f, Cxy = scipy.signal.coherence(data_axswap[i][k], data_axswap[j][k],
                                        fs=fs, window=window, noverlap=noverlap)
        msc_mat[i, j] += Cxy

  f_step = fs/2/192
  i_lh = np.int(np.ceil(band[0]/f_step))
  i_hl = np.int(np.floor(band[1]/f_step))
  wl = (f[i_lh]-band[0])/f_step
  wh = (band[1]-f[i_hl])/f_step

  msc_band = msc_mat[..., i_lh-1]*wl + \
  np.sum(msc_mat[..., i_lh:i_hl], axis=-1) + \
  msc_mat[..., i_hl]*wh
  msc_band = msc_band / (i_hl-i_lh)+wl+wh

  return msc_band

  
def erdos_renyi(N, p, connect=True, maxit=100):
  '''
  erdos_renyi random graph.
  TODO: implement connect, maxit
  '''
#  density = p / (2.0 - 1.0/N)
  A = scipy.sparse.rand(N, N, density=p, format='coo').todense()
  A = (A + A.transpose())/2 
  A[np.diag_indices(N)] = 0
  A = np.where(A > 0, np.float32(1), np.float32(0))
  return A
  
  
def adjacency_dist_single_layer(dist_metric, k=-1, sigma=None, p=None):
  channel_dim = 32
  d = get_distance_matrix()

  # pdist: m by n array of m observation of n-dim space
  # 32x3
  if sigma is None:
    sigma = np.var(d)
    
  d = np.exp(-(np.square(d)/(2*sigma)))
  
  if dist_metric == 'geometric':
    if p is None:
      p = np.percentile(np.flatten(d), 50)
    d[d >= p] = 0
  else:
    # k-NN graph.
    if k < 0:
      k = 32

    d[np.diag_indices(channel_dim)] = 0
    idx = np.argsort(d, axis=1)[:, :-k]
    ir = np.mgrid[0:channel_dim, 0:(channel_dim-k)][0]
    d[ir, idx] = 0
  
  d = (d + np.transpose(d))/2 # to guarantee symmetricity

  d[np.diag_indices(channel_dim)] = 0

  return d


def stack_graph_layers(g, n_stack, connect_same=False):
  g_dim = g.shape[0]
  new_dim = g_dim * n_stack

  new_g = np.zeros((new_dim, new_dim))

  same_graph = np.eye(g_dim)

  for i in range(n_stack):
    for j in range(n_stack):
      if i == j:
        new_g[i*g_dim:(i+1)*g_dim, j*g_dim:(j+1)*g_dim] = g
      else:
        if connect_same:
          new_g[i*g_dim:(i+1)*g_dim, j*g_dim:(j+1)*g_dim] = same_graph

  return new_g

  
def edge_to_vertex_signal(connectivity, N=None):
  '''
  Make new graph whose vertices are edges of the original (undirected) graph.
  This function returns signal.
  
  connectivity - NxN symmetric matrix
  '''
  
  if N is None:
    N = connectivity.shape[0]
  
  connectivity = np.moveaxis(connectivity, -1, -2)
  row, col = np.triu_indices(N, k=1)
  C_signal = connectivity[..., row, col]
  
  return C_signal

  

def edge_to_vertex_graph(N=32, dist_type='geometric', dist_metric='euclidean', max_k=-1, graph=None):
  '''
  Make new graph whose vertices are edges of the original (undirected) graph.
  Graph is based on electrode positions of EEG.
  '''      
  E = np.int32(N*(N-1)//2)
  Ge = np.zeros((E, E))
  
  # distance measure type 1 - connecting connectivity via node
  # edge - node - edge --> edge in edge graph
  # Than, what is suitable for the edge weight?
  #  1. averaging two edge weights
  #    - edge consists of two stubs.
  #    - the edge node (a,b) can be positioned at the center of a and b. (two stubs)
  #  2. property of the sharing node
  #    - signal power of node-level signal?
  if dist_type == 'share':
    if graph is None:
      return None
      
    if scipy.sparse.issparse(graph):
      G = np.triu(graph.toarray())
    else:
      G = np.triu(graph)
      
    for i in np.range(N):
      e = np.where(G[i])
      for j, k in itertools.combinations(range(len(e)), 2):
        val = (G[i,j]+G[i,k])/2
        Ge[N*i+e[j], N*i+e[k]] = val
        Ge[N*i+e[k], N*i+e[j]] = val
  elif dist_type == 'geometric':
    A = get_distance_matrix('euclidean')
    Au = np.triu(A)
    
    def divide(n, a):
      if n < 31-a:
        return n, a
      else:
        return divide(n-(31-a), a+1)
        
    for i in range(E):
      i2, i1 = divide(i, 0)
      for j in range(i+1, E):
        j2, j1 = divide(j, 0)
        dist_min = np.min([Au[i1, j1], Au[i1, j2], Au[i2, j1], Au[i2, j2]])
        Ge[i, j] = dist_min + 0.5*(Au[i1, i2] + Au[j1, j2])
        
    Ge = Ge + np.transpose(Ge)
    Ge = np.exp(-(np.square(Ge)/(2*np.var(Ge))))
    Ge[np.diag_indices(E)] = 0
    
    if max_k > 0:
      idx = np.argsort(Ge, axis=1)[:, :-max_k]
      ir = np.mgrid[0:E, 0:(E-max_k)][0]
      Ge[ir, idx] = 0
    
  return Ge
    
    
  # dist((a,b),(c,d)) = 
  #  0 if (a,b) == (c,d)
  #  min(dist(a,c),dist(a,d),dist(b,c),dist(b,d))+0.5(dist(a,b)+dist(c,d))
  

def conn_distance():
  with open('dist_adjacency.dat', 'rb') as f:
    A = pickle.load(f)
  return np.asarray(A.todense())