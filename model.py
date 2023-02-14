import torch
import numpy as np
from torchvision import transforms, utils
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union, List, Optional
from torch_geometric.typing import OptTensor
from torch_geometric.data import Data

class FeatureExtractor(torch.nn.Module):
    '''
    Feature Extractor class
    '''
    def __init__(self, name = None, weights = None) -> None:
        super().__init__()
        
        name = name.lower()
        all_models = ['alexnet', "convnext", "densenet", "efficientnet", "efficientnetv2",
                        "googlenet", "inceptionv3", "maxvit", "mnasnet", "mobilenetv2", "mobilenetv3",
                        "regnet", "resnet", "resnext", "shufflenetv2", "squeezenet", "shufflenetv2",
                        "swin", "vgg", "vit", "wideresnet"]
        assert name.lower() in all_models, """Supported models are ['AlexNet', "ConvNext", "DenseNet", "EfficientNet", "EfficientNetV2",
                        "GoogLeNet", "InceptionV3", "MaxVit", "MNASNet", "MobileNetV2", "MobileNetV3",
                        "RegNet", "ResNet", "ResNeXt", "ShuffleNetV2", "SqueezeNet", "ShuffleNetV2",
                        "SWIN", "VGG", "ViT", "WideResNet"]"""
        
        #Feature extractor 
        self.fx_base = self.load_model(name, weights)     #resnet50(weights = 'IMAGENET1K_V2')       #feature extractor model
        
        for param in self.fx_base.parameters():
            param.requires_grad = False   #so that with torch.no_grad is not required
        self.fx_required = {'flatten':'flatten'}        #required layers from all the get_graph_node_names(m)
        self.fx = create_feature_extractor(self.fx_base, return_nodes=self.fx_required)       #last dense layer

    def load_model(self, name, weights = None):
        m = MODELS()
        return m.load_model(name)

    def forward(self, x):
        return self.fx(x)['flatten']

class Matcher(torch.nn.Module):
    '''
    Feature matcher class to compute similarity of given two node feature
    '''    
    def forward(self, n1, n2, edge_index = None):
        # n1 = n1[edge_index[0]]
        # n2 = n2[edge_index[1]]
        return (n1 @ n2.t()) / (torch.norm(n1, dim =-1).unsqueeze(-1) @ torch.norm(n2, dim =-1).unsqueeze(-1).t())


class LoopClosureDetector(torch.nn.Module):
    '''
    LCD base model for extracting features of the node image & then calculating matching criteria to
    predict edge between two nodes.
    '''
    def __init__(self, FX, MX) -> None:
        super().__init__()  
        self.fx = FX
        self.mx = MX

    def forward(self, x):
        # x has shape [N, in_channels]  
        # edge_index has shape [2, E]
        # introduce another variable and then make the comparision bewteen them
        # 1st -> graph nodes, 2nd-> another graph nodes or indiviudal image to add in graph
        x = self.fx(x)
        pred = self.mx(x, x)
        return pred
    
class MODELS:
    
    def load_model(self, name):
        default = self.resnet
        return getattr(self, name, lambda: default)()

    def alexnet(self):
        from torchvision.models import AlexNet
        return AlexNet(weights = 'DEFAULT')

    def convnext(self):
        from torchvision.models import ConvNeXt
        return ConvNeXt(weights = 'DEFAULT')

    def densenet(self):
        from torchvision.models import DenseNet
        return DenseNet(weights = 'DEFAULT')

    def efficientnet(self):
        from torchvision.models import EfficientNet
        return EfficientNet(weights = 'DEFAULT')

    def efficientnetv2(self):
        from torchvision.models import efficientnet_v2_l
        return efficientnet_v2_l(weights = 'DEFAULT')

    def googlenet(self):
        from torchvision.models import GoogLeNet
        return GoogLeNet(weights = 'DEFAULT')

    def inceptionv3(self):
        from torchvision.models import Inception3
        return Inception3(weights = 'DEFAULT')

    def maxvit(self):
        from torchvision.models import MaxVit
        return MaxVit(weights = 'DEFAULT')

    def mnasnet(self):
        from torchvision.models import MNASNet
        return MNASNet(weights = 'DEFAULT')

    def mobilenetv2(self):
        from torchvision.models import MobileNetV2
        return MobileNetV2(weights = 'DEFAULT')

    def mobilenetv3(self):
        from torchvision.models import MobileNetV3
        return MobileNetV3(weights = 'DEFAULT')

    def regnet(self):
        from torchvision.models import RegNet
        return RegNet(weights = 'DEFAULT')
    
    def resnet(self):
        from torchvision.models import resnet50
        return resnet50(weights = 'IMAGENET1K_V2')
        
    def resnext(self):
        from torchvision.models import resnext50_32x4d
        return resnext50_32x4d(weights = 'DEFAULT')

    def shufflenetv2(self):
        from torchvision.models import ShuffleNetV2
        return ShuffleNetV2(weights = 'DEFAULT')

    def squeezenet(self):
        from torchvision.models import SqueezeNet
        return SqueezeNet(weights = 'DEFAULT')

    def swin(self):
        from torchvision.models import SwinTransformer
        return SwinTransformer(weights = 'DEFAULT')

    def vgg(self):
        from torchvision.models import vgg19
        return vgg19(weights = 'DEFAULT')

    def vit(self):
        from torchvision.models import VisionTransformer
        return VisionTransformer(weights = 'DEFAULT')

    def wideresnet(self):
        from torchvision.models import wide_resnet50_2
        return wide_resnet50_2(weights = 'DEFAULT')


class customGraph(Data):

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
    @property
    def num_nodes(self) -> Optional[int]:
        try:
            return len(self.x)
        except:
            # setattr(self, 'num_nodes', 0)
            return 0

    def add_node(self, x:OptTensor, edges: Union[List, None] = None, edge_attr: Union[List, None] = None,
                 pos: OptTensor = None):
        
        if self.num_nodes == 0:
            '''
            Intializing graph from the beginning.
            '''
            assert edges == None, "New Graph initializing from single node can't have edges."
            node_id = self.num_nodes
            self.x = x
            self.node_id = [node_id]
        else:
            node_id = self.num_nodes
            self.x = torch.cat((self.x, x))
            self.node_id.append(node_id)

            if edges!=None and edge_attr!=None:
                no_edges = len(edges)
                assert (torch.tensor(edges) < self.num_nodes).all, "Only Existing node edges can be added."
                u = [node_id]*no_edges
                
                #add edges
                edges = torch.tensor(np.array([u,edges]), dtype = torch.long)
                edge_attr = torch.tensor(np.array(edge_attr).reshape(-1,1))
                return self.add_edge(edges, edge_attr)
        return True    

    def add_edge(self, edge_index, edge_attr):
        assert edge_index.ndim == 2
        assert edge_index.shape[0] == 2
        self.edge_index = torch.cat((self.edge_index, edge_index), dim = -1) if self.edge_index!=None else edge_index


        # add edge attr
        assert edge_index.ndim == 2
        assert edge_attr.shape[0] == edge_index.shape[1]        #match number of nodes
        if self.edge_attr!=None:
            assert edge_attr.shape[1] == self.edge_attr.shape[1]
            self.edge_attr = torch.cat((self.edge_attr, edge_attr))
        else:
            self.edge_attr = edge_attr
        return True
        

    def get_node(self, id: int):
        return self.x[id].detach().cpu().permute(1,2,0).numpy()

    def get_edge(self, u:int, v:int):
        '''
        return edge attribute for the source node u, to destination node v edge.
        '''
        assert self.edge_attr is not None, "No Egde atributes set."
        pass