import os
import time
import wandb
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.data import makedirs, extract_zip

from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
from model import FeatureExtractor, Matcher
# from utils import dense_matrics
# from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

default_config = SimpleNamespace(
    project_name = 'LCD',
    dataset_name = 'NC_reduced',
    model_name = 'ResNet',
    weights = 'default'
)

def pass_args():
    "Overriding default args"
    argparser = argparse.ArgumentParser(description='Loop closing daataset & model details')
    argparser.add_argument("--project_name", type=str, default = default_config.project_name, help='Project name to log data in W&B.')
    argparser.add_argument("--dataset_name", type=str, default = default_config.dataset_name, help='Dataset to load.',
                            choices= ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTY_00', 'KITTY_05',
                                        'KAIST_NORTH','KAIST_EAST','KAIST_WEST'])
    argparser.add_argument("--model_name", type=str, default = default_config.model_name, help='Model to use for LCD.',
                            choices=['AlexNet', "ConvNext", "DenseNet", "EfficientNet", "EfficientNetV2",
                                        "GoogLeNet", "InceptionV3", "MaxVit", "MNASNet", "MobileNetV2", "MobileNetV3",
                                        "RegNet", "ResNet", "ResNeXt", "ShuffleNetV2", "SqueezeNet", "ShuffleNetV2",
                                        "SWIN", "VGG", "ViT", "WideResNet"])
    argparser.add_argument("--weights", type=str, default = default_config.weights, help='Weight path/name to load on model.')  #ignoring for now


def train(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ######################################################################################
    # Download dataset & extract features
    load_dataset(name = config.dataset_name)
    run = wandb.init(project = config.project_name, entity = "abhi_khoyani", config = config)

    
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
    for batch in dataloader:
        batch = batch.to(device)
        f = fx(batch)
        xx.append(f.detach().cpu())

    
    data = Data()
    data.node_id = torch.arange(len(gt))
    u, v  = np.where(np.ones((data.num_nodes, data.num_nodes)))  #adding edgegs between all the nodes
    data.edge_index = torch.tensor(np.array([u,v]), dtype = torch.long)
    data.edge_attr = torch.tensor(np.expand_dims(gt.flatten(),-1), dtype = torch.bool)
    data.x = torch.cat(xx, dim = 0)


    ######################################################################################
    
    mx = Matcher()
    pred = mx(data.x, data.x)

    #closing timer
    end_time = time.perf_counter()

    
    #all data to gpu
    # data = data.to(device)
    # model = model.eval().to(device)

    # with torch.autograd.profiler.profile() as prof:
        #calculate predictions
        # with torch.no_grad():
            # pred = model(data.x)
        # pred = pred.detach().cpu().numpy()

    # edge_index = data.edge_index.detach().cpu().numpy()
    # gt = data.edge_attr.detach().cpu().numpy().squeeze()
        
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

    all_dataset = ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTY_00', 'KITTY_05',
                'KAIST_NORTH','KAIST_EAST','KAIST_WEST']
    assert name in all_dataset, """Supported dataset are ['CC_orig', 'NC_orig', 'CC_reduced', 'NC_reduced', 'KITTY_00', 'KITTY_05',
                                            'KAIST_NORTH','KAIST_EAST','KAIST_WEST']"""

    assert os.path.isdir(URL), "Invalid BASE_DIR provided"

    root_dir = os.path.join(URL, name)
    if not os.path.isdir(root_dir):
        print('Downloading Dataset....')
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

