import os
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import default_collate
from torch_geometric.data import Data
from torch_geometric.data import makedirs, extract_zip

def load_dataset(name = None, URL = './Datasets'):

    all_dataset = ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTY_00', 'KITTY_05',
                'KAIST_NORTH','KAIST_EAST','KAIST_WEST']
    assert name in all_dataset, """Supported dataset are ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTY_00', 'KITTY_05',
                                            'KAIST_NORTH','KAIST_EAST','KAIST_WEST']"""

    assert os.path.isdir(URL), "Invalid BASE_DIR provided"

    root_dir = os.path.join(URL, name)
    image_path = os.path.join(root_dir, 'Image')
    if not os.path.isdir(root_dir):
        print('Downloading Dataset....')
        makedirs(root_dir)
        extract_zip(os.path.join(URL, name+'.zip'), root_dir)
        extract_zip(os.path.join(URL, name, 'Image.zip'), root_dir)
        print('Downloading Finished....')

    gt = loadmat(os.path.join(root_dir, 'gt.mat'))['truth'].astype(bool)    #storing it in boolean for memory saving
    gt = gt + gt.T

    data = Data()
    data.node_id = torch.arange(len(gt))
    u, v  = np.where(np.ones((data.num_nodes, data.num_nodes)))  #adding edgegs between all the nodes
    data.edge_index = torch.tensor(np.array([u,v]), dtype = torch.long)
    data.edge_attr = torch.tensor(np.expand_dims(gt.flatten(),-1), dtype = torch.bool)

    xx = []
    for i in sorted(os.listdir(image_path)):
        xx.append(transforms.ToTensor()(Image.open(os.path.join(image_path, i))))

    data.x = default_collate(xx)

    return data
