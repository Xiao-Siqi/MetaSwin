import torch
from typing import Union
from torchvision.models.feature_extraction import get_graph_node_names

from .pim_module import pim_module

"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""

def load_model_weights(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def build_swintransformer(pretrained: bool = True,
                          num_selects: Union[dict, None] = None,
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          metainfo_dims: int = [2,3],
                          comb_type : str = "GCN_2",
                          comb_proj_size: Union[int, None] = None):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()

    print("Building...")
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = None,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects,
                                   use_combiner = num_selects,
                                   metainfo_dims = metainfo_dims,
                                   comb_type = comb_type,
                                   comb_proj_size = comb_proj_size)




if __name__ == "__main__":
    ### ==== resnet50 ====
    # model = build_resnet50(pretrained='./resnet50_miil_21k.pth')
    # t = torch.randn(1, 3, 448, 448)

    ### ==== swin-t ====
    # model = build_swintransformer(False)
    # t = torch.randn(1, 3, 384, 384)

    ### ==== vit ====
    # model = build_vit16(pretrained='./vit_base_patch16_224_miil_21k.pth')
    # t = torch.randn(1, 3, 448, 448)

    ### ==== efficientNet ====
    model = build_efficientnet(pretrained=False)
    t = torch.randn(1, 3, 448, 448)

    model.cuda()

    t = t.cuda()
    outs = model(t)
    for out in outs:
        print(type(out))
        print("    " , end="")
        if type(out) == dict:
            print([name for name in out])


MODEL_GETTER = {
    "swin-t":build_swintransformer
}
