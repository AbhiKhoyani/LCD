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
        print(f'Loading model: {name}')
        return getattr(self, name, lambda: default)()

    def alexnet(self):
        from torchvision.models import alexnet
        fx_required = {'flatten':'out'}
        base = alexnet(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def convnext(self):
        from torchvision.models import convnext_base
        fx_required = {'classifier.1':'out'}
        base = convnext_base(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def densenet(self):
        from torchvision.models import densenet161
        fx_required = {'flatten':'out'}
        base = densenet161(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def efficientnet(self):
        from torchvision.models import efficientnet_b6
        fx_required = {'flatten':'out'}
        base = efficientnet_b6(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def googlenet(self):
        from torchvision.models import googlenet
        fx_required = {'flatten':'out'}
        base = googlenet(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def inceptionv3(self):
        from torchvision.models import inception_v3
        fx_required = {'flatten':'out'}
        base = inception_v3(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def maxvit(self):
        from torchvision.models import maxvit_t
        fx_required = {'flatten':'out'}
        base = maxvit_t(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def mnasnet(self):
        from torchvision.models import mnasnet1_3
        fx_required = {'mean':'out'}
        base = mnasnet1_3(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def mobilenetv2(self):
        from torchvision.models import mobilenet_v2
        fx_required = {'flatten':'out'}
        base = mobilenet_v2(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def mobilenetv3(self):
        from torchvision.models import mobilenet_v3_large
        fx_required = {'flatten':'out'}
        base = mobilenet_v3_large(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def regnet(self):
        from torchvision.models import regnet_y_32gf
        fx_required = {'flatten':'out'}
        base = regnet_y_32gf(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)
    
    def resnet(self):
        from torchvision.models import resnet50
        fx_required = {'flatten':'out'}
        base = resnet50(weights = 'IMAGENET1K_V2')
        return create_feature_extractor(base, return_nodes=fx_required)
        
    def resnext(self):
        from torchvision.models import resnext101_64x4d
        fx_required = {'flatten':'out'}
        base = resnext101_64x4d(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def shufflenetv2(self):
        from torchvision.models import shufflenet_v2_x2_0
        fx_required = {'mean':'out'}
        base = shufflenet_v2_x2_0(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def squeezenet(self):
        from torchvision.models import squeezenet1_1
        fx_required = {'flatten':'out'}
        base = squeezenet1_1(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def swin(self):
        from torchvision.models import swin_v2_s
        fx_required = {'flatten':'out'}
        base = swin_v2_s(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def vgg(self):
        from torchvision.models import vgg16_bn
        fx_required = {'flatten':'out'}
        base = vgg16_bn(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def vit(self):
        from torchvision.models import vit_b_16
        fx_required = {'getitem_5':'out'}
        base = vit_b_16(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)

    def wideresnet(self):
        from torchvision.models import wide_resnet101_2
        fx_required = {'flatten':'out'}
        base = wide_resnet101_2(weights = 'DEFAULT')
        return create_feature_extractor(base, return_nodes=fx_required)


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