import wandb
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from load_dataset import load_dataset
from model import LoopClosureDetector, FeatureExtractor, Matcher
from utils import dense_matrics
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

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
    
    data = load_dataset(config.dataset_name)
    fx = FeatureExtractor(name = config.model_name) #not supporting weights args as of now
    mx = Matcher()
    model = LoopClosureDetector(fx, mx)

    run = wandb.init(project = config.project_name, entity = "abhi_khoyani", config = config)
    #all data to gpu
    # data = data.to(device)
    # model = model.eval().to(device)

    with torch.autograd.profiler.profile() as prof:
        #calculate predictions
        with torch.no_grad():
            pred = model(data.x)
        pred = pred.detach().cpu().numpy()

    edge_index = data.edge_index.detach().cpu().numpy()
    gt = data.edge_attr.detach().cpu().numpy().squeeze()
        
    #logging gt & pred in confusion matrix form
    # plt.imshow(gt)
    wandb.log({"Ground Truth": wandb.Image(dense_matrics(edge_index, gt))})
    # plt.imshow(pred)
    wandb.log({"Prediction": wandb.Image(pred*255)})

    #evaluation
    gt = np.expand_dims(gt.flatten(), -1)
    pred = np.expand_dims(pred.flatten(), -1)
    pred =  np.vstack([1 - pred[:,0], pred[:,0]]).T
    wandb.log({"PR curve":wandb.plot.pr_curve(gt, pred, labels = [0,1])})
    wandb.run.summary['CPU total time'] = prof.self_cpu_time_total
    
    run.finish()


if __name__ == "__main__":
    pass_args()
    train(default_config)

