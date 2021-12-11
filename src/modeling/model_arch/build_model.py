from omegaconf import DictConfig

from src.modeling.model_arch.conv_models import LuxNet, LuxResNet
from src.modeling.model_arch.image_caption import SimpleLSTMModel


def build_model_from_conf(conf: DictConfig):
    net_kwargs = {
        "in_channels": conf.obs.num_state_features,
        "out_channels": conf.action.num_actions,
    }
    type_conf = conf.model[conf.model.type]

    print(f"using {conf.model.type}")
    if conf.model.type == "imitation_baseline":
        net_class = LuxNet
        # net_kwargs.update(
        #     {
        #         "block_type": type_conf.block_type
        #     }
        # )
    elif conf.model.type == "resnet_baseline":
        net_class = LuxResNet
    elif conf.model.type == "image_caption":
        net_kwargs.update(
            {
                "timm_params": conf.model.timm_params,
                "gem_power": conf.model.pool.gem_power,
                "gem_requires_grad": conf.model.pool.gem_requires_grad,
                "num_classes": -1,
                "encoder": "not-imagenet",
                "encoder_out_indices": type_conf.encoder_out_indices,
                "encoder_hidden_dim": type_conf.encoder_hidden_dim,
                "in_channels": type_conf.in_channels,
                "decoder": conf.model.decoder,
                "use_point_conv": False
                if not hasattr(type_conf, "use_point_conv")
                else type_conf.use_point_conv,
            }
        )
        net_class = SimpleLSTMModel

    else:
        raise NotImplementedError(f"model.type : {conf.model.type}")
    return net_kwargs, net_class
