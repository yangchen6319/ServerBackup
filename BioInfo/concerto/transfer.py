import os
import sys
sys.path.append("../")
from concerto_function5_3 import *
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

#Select an available GPU to run on a multi-GPU computer or you can run it directly on the CPU without executing this cell
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 

path = './data/hp_rmSchwann_commonCelltype.loom'
adata = sc.read(path)

# 数据处理
adata = preprocessing_rna(adata,n_top_features=2000,is_hvg=True,batch_key='tech')

adata_ref = adata[adata.obs['tech'] != 'indrop']
adata_query = adata[adata.obs['tech'] == 'indrop']
save_path = './data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
adata_ref.write_h5ad(save_path + 'adata_ref.h5ad')
adata_query.write_h5ad(save_path + 'adata_query.h5ad')
