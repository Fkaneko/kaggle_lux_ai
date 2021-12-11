from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.modeling.model_arch.build_model import build_model_from_conf
from src.rl.internal_validation import (
    ImageCaptionAgent,
    internal_match,
    load_baseline_model,
    load_opponent,
)
from src.rl.model import PolicyValueNet
from src.utils.util import set_random_seed


def to_numpy(tensor):
    " ""
    from pytorch documentation
    https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    """
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def generate_image_caption_agent(
    conf: DictConfig, rl_model_path: str, use_onnx: bool = False
):
    net_kwargs, net_class = build_model_from_conf(conf=conf)

    model = net_class(**net_kwargs)

    rl_model = PolicyValueNet(
        encoder=model.encoder,
        decoder=model.decoder,
    )
    rl_model.load_state_dict(torch.load(rl_model_path), strict=True)
    rl_model = rl_model.eval()

    if use_onnx:
        print("using onnx")
        export_path = "./model.onnx"
        # from pytorch documentation
        # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
        example_input = {
            "image": torch.rand(1, 20, 32, 32).clamp(0.0, 1.0),
            "input_sequence": torch.rand(1, 128, 4).clamp(0.0, 0.31),
        }
        torch_out = rl_model(example_input)
        torch.onnx.export(
            rl_model.cpu(),
            (example_input),
            f=export_path,
            export_params=True,
            input_names=list(example_input.keys()),
            output_names=["policy", "value"],
            opset_version=11,
        )
        model = onnx.load(export_path)
        # Check that the model is well formed
        onnx.checker.check_model(model)
        # Print a human readable representation of the graph
        # print(onnx.helper.printable_graph(model.graph))

        ort_session = ort.InferenceSession(export_path)
        ort_inputs = {
            ort_in.name: to_numpy(example_input[ort_in.name])
            for ort_in in ort_session.get_inputs()
        }

        ort_outs = ort_session.run(
            None,
            ort_inputs,
        )

        # compare ONNX Runtime and PyTorch results
        for i, key in enumerate(torch_out.keys()):
            np.testing.assert_allclose(
                to_numpy(torch_out[key]), ort_outs[i], rtol=1e-03, atol=1e-05
            )
        rl_model = ort_session

    player = ImageCaptionAgent(
        mode="pred",
        model=rl_model,
    )

    return player


if __name__ == "__main__":
    set_random_seed()
    player_model = load_baseline_model()
    opponent = load_opponent(
        path="../input/lux_ai_baseline_imitation_weight/lb_1350"
        # path="../input/lux_ai_baseline_imitation_weight"
    )

    with initialize(config_path="./src/config"):
        conf = compose(config_name="config")
        print(OmegaConf.to_yaml(conf))

    epoch = 100
    rl_model_path = f"./models/{epoch}.pth"
    player = generate_image_caption_agent(conf, rl_model_path, use_onnx=True)

    num_episodes = 13
    map_sizes = [12, 16, 24, 32]
    for map_size in map_sizes:
        res = internal_match(
            player,
            opponent,
            num_episodes=num_episodes,
            seeds=list(range(1, num_episodes + 1)),
            map_size=map_size,
            replay_folder=f"../working/replay_kaggle_debug_{map_size}_{epoch}",
            replay_stateful=False,
            vis_results=True,
        )
        print(res)
