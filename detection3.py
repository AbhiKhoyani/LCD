'''
This code loades each image one by one from Imageloader, calculates its feature and add it to the graph by
creating a new node. Then it will calculate the matching score with all the existing graph node and add it
to its edge_attr. This code simulates exact scenario of the runtime loop closing detection.

After the completion, we'll get half traingle only in predictions so, adding transpose with it to get full
square confusion matrix.
'''

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
from model import FeatureExtractor, Matcher, customGraph
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from utils import dense_matrics

default_config = SimpleNamespace(
    project_name = 'Test',
    dataset_name = 'NC_reduced',
    model_name = 'ShuffleNetV2',
    weights = 'default'
)

def pass_args():
    "Overriding default args"
    argparser = argparse.ArgumentParser(description='Loop closing daataset & model details')
    argparser.add_argument("--project_name", type=str, default = default_config.project_name, help='Project name to log data in W&B.')
    argparser.add_argument("--dataset_name", type=str, default = default_config.dataset_name, help='Dataset to load.',
                            choices= ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                                        'KAIST_North','KAIST_East','KAIST_West'])       #9
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
    
    if config.dataset_name in ['CC_orig', 'NC_orig', 'KITTI_00', 'KITTI_05']:
        gt = gt.T + np.eye(len(gt), dtype = np.bool_)
    # else:
    #     gt = gt #+ gt.T

    dataset = ImageFolderDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4) 
    fx = FeatureExtractor(name = config.model_name).eval().to(device)
    mx = Matcher()
    graph = customGraph().to(device)
    
    #starting timer
    start_time = time.perf_counter()
    for batch in tqdm.tqdm(dataloader):
        batch = batch.to(device)
        internal_start_time = time.perf_counter()
        f = fx(batch)
        graph.add_node(f)
        match_score = mx(graph.x , f).detach().cpu()
        edge_index = torch.tensor([graph.node_id,[graph.num_nodes-1]*(graph.num_nodes)]).to(device)
        graph.add_edge(edge_index,match_score)
        internal_end_time = time.perf_counter()
        wandb.log({'step-timer':internal_end_time - internal_start_time})

    ######################################################################################
    pred = dense_matrics(graph.edge_index.detach().cpu().numpy().astype(int), graph.edge_attr.detach().cpu().numpy())
    pred = pred #+ pred.T - np.eye(len(pred))

    #closing timer
    end_time = time.perf_counter()

        
    # logging gt & pred in confusion matrix form
    # plt.imshow(gt)
    # wandb.log({"Ground Truth": wandb.Image(dense_matrics(edge_index, gt))})
    wandb.log({"Ground Truth": wandb.Image(gt)})
    # plt.imshow(pred)
    wandb.log({"Prediction": wandb.Image(pred*255)})

    # mask for upper triangle to consider only valid values and ignoring all zeros
    mask = np.triu(np.ones((len(gt), len(gt)))).astype(np.bool_)  
    pred = pred[mask]
    gt = gt[mask]
    precision, recall, _ = precision_recall_curve(gt, pred)
    wandb.run.summary['AUC'] = auc(recall, precision)
    wandb.run.summary['Avg PR'] = average_precision_score(gt, pred)
    wandb.run.summary['PR @ Recall_0'] = max(precision[recall == 0])
    wandb.run.summary['Recall @ Pr_100'] = max(recall[precision == 1])

    # evaluation
    gt = np.expand_dims(gt, -1)
    pred = np.expand_dims(pred, -1)
    pred = np.vstack([1 - pred[:,0], pred[:,0]]).T
    wandb.log({"PR curve":wandb.plot.pr_curve(gt, pred, labels = [0,1])})
    wandb.run.summary['Total time'] = end_time - start_time
    wandb.run.summary['FPS'] = len(gt)/(end_time - start_time)
    


    run.finish()


def load_dataset(name = None, URL = './Datasets'):

    all_dataset = ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                'KAIST_North','KAIST_East','KAIST_West']
    assert name in all_dataset, """Supported dataset are ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTI_00', 'KITTI_05',
                                            'KAIST_North','KAIST_East','KAIST_West']"""

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