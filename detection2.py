import os
import time
import wandb
import tqdm
import argparse
import torch
import numpy as np
from types import SimpleNamespace

from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.loader import  NeighborLoader
from torch_geometric.data import makedirs, extract_zip

# from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
from model import FeatureExtractor, Matcher
from utils import dense_matrics

default_config = SimpleNamespace(
    project_name = 'Test',
    dataset_name = 'NC_reduced',
    model_name = 'ResNet',
    weights = 'default'
)

def pass_args():
    "Overriding default args"
    argparser = argparse.ArgumentParser(description='Loop closing daataset & model details')
    argparser.add_argument("--project_name", type=str, default = default_config.project_name, help='Project name to log data in W&B.')
    argparser.add_argument("--dataset_name", type=str, default = default_config.dataset_name, help='Dataset to load.',
                            choices= ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                                        'KAIST_NORTH','KAIST_EAST','KAIST_WEST'])       #9
    argparser.add_argument("--model_name", type=str, default = default_config.model_name, help='Model to use for LCD.',
                            choices=['AlexNet', "ConvNext", "DenseNet", "EfficientNet", "EfficientNetV2",
                                        "GoogLeNet", "InceptionV3", "MaxVit", "MNASNet", "MobileNetV2", "MobileNetV3",
                                        "RegNet", "ResNet", "ResNeXt", "ShuffleNetV2", "SqueezeNet",
                                        "SWIN", "VGG", "ViT", "WideResNet"])            #20
    argparser.add_argument("--weights", type=str, default = default_config.weights, help='Weight path/name to load on model.')  #ignoring for now
    args = argparser.parse_args()
    vars(default_config).update(vars(args))


def train(config):    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ######################################################################################
    # Download dataset & extract features
    load_dataset(name = config.dataset_name)
    run = wandb.init(project = config.project_name, entity = "abhi_khoyani", config = config,
                         name=config.model_name)

    root_dir = os.path.join('./Datasets', config.dataset_name)
    image_path = os.path.join(root_dir, 'Image')
    gt = loadmat(os.path.join(root_dir, 'gt.mat'))['truth'].astype(bool)    #storing it in boolean for memory saving
    gt = gt + gt.T

    dataset = ImageFolderDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4) 
    fx = FeatureExtractor(name = config.model_name).eval().to(device)

    #starting timer
    start_time = time.perf_counter()

    xx = []
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     with torch.no_grad():

    for batch in tqdm.tqdm(dataloader):
        internal_start_time = time.perf_counter()
        batch = batch.to(device)
        f = fx(batch)
        xx.append(f.detach().cpu())
        internal_end_time = time.perf_counter()
        wandb.log({'fx_timer':internal_end_time - internal_start_time})
    
    data = Data()
    data.node_id = torch.arange(len(gt))
    u, v  = np.where(np.ones((data.num_nodes, data.num_nodes)))  #adding edgegs between all the nodes
    data.edge_index = torch.tensor(np.array([u,v]), dtype = torch.long)
    data.edge_attr = torch.tensor(np.expand_dims(gt.flatten(),-1), dtype = torch.bool)
    data.x = torch.cat(xx, dim = 0)


    ######################################################################################
    
    mx = Matcher()
    # pred = mx(data.x, data.x)

    #real time scenario with Neighborloader
    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1, time_attr='node_id', shuffle = False)
    source = []
    dest = []
    predictions = []
    for b in tqdm.tqdm(loader):
        '''
        Here comes each nodes and its neighbors.
        1. we have features stored for each nodes, run matcher and store final value in form of u,v
        '''
        internal_start_time = time.perf_counter()
        n1 = b.x[b.input_id].to(device)
        n2 = b.x[b.node_id].to(device)
        
        pred = mx(n1, n2).flatten()
        u = torch.cat((b.input_id, b.node_id[b.edge_index[0]])).numpy()
        v = torch.cat((b.input_id, b.node_id[b.edge_index[1]])).numpy()

        internal_end_time = time.perf_counter()
        wandb.log({'mx_timer':internal_end_time - internal_start_time})
        source.append(u)
        dest.append(v)
        predictions.append(pred.detach().cpu())

    predictions = torch.cat(predictions)
    source = np.concatenate(source)
    dest = np.concatenate(dest)
    pred = dense_matrics(np.array((source, dest)), predictions)
    pred = (pred + pred.T)/2

    #closing timer
    end_time = time.perf_counter()

        
    #logging gt & pred in confusion matrix form
    # plt.imshow(gt)
    # wandb.log({"Ground Truth": wandb.Image(dense_matrics(edge_index, gt))})
    wandb.log({"Ground Truth": wandb.Image(gt)})
    # plt.imshow(pred)
    wandb.log({"Prediction": wandb.Image(pred*255)})

    #evaluation
    gt = np.expand_dims(gt.flatten(), -1)
    pred = np.expand_dims(pred.flatten(), -1)
    pred =  np.vstack([1 - pred[:,0], pred[:,0]]).T
    wandb.log({"PR curve":wandb.plot.pr_curve(gt, pred, labels = [0,1])})
    wandb.run.summary['Total time'] = end_time - start_time
    
    run.finish()


def load_dataset(name = None, URL = './Datasets'):

    all_dataset = ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                'KAIST_NORTH','KAIST_EAST','KAIST_WEST']
    assert name in all_dataset, """Supported dataset are ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                                            'KAIST_NORTH','KAIST_EAST','KAIST_WEST']"""

    assert os.path.isdir(URL), "Invalid BASE_DIR provided"

    root_dir = os.path.join(URL, name)
    if not os.path.isdir(root_dir):
        print('Downloading Dataset....',name)
        makedirs(root_dir)
        extract_zip(os.path.join(URL, name+'.zip'), root_dir)
        extract_zip(os.path.join(URL, name, 'Image.zip'), root_dir)
        print('Downloading Finished....')
    return

class ImageFolderDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = sorted(os.listdir(data_dir))
        self.transform = transforms.Compose([
                            transforms.ToTensor()
                        ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = Image.open(image_path)
        image = self.transform(image)
        return image


if __name__ == "__main__":
    pass_args()
    train(default_config)

