import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from models import Network


def plot_mlp(model: Network) -> None:
    layer_size_list = model.layer_size_list
    layer_list = model.layer_list
    depth = model.depth + 1

    _, ax = plt.subplots(figsize=(3, 3))
    s = 1 / (2 * max(layer_size_list))

    for layer_index in range(depth):
        neuron_num = layer_size_list[layer_index]
        for neuron_index in range(neuron_num):
            circle = Ellipse(
                (
                    1 / (2 * neuron_num) + neuron_index / neuron_num,
                    0.1 * (depth - 1 - layer_index),
                ),
                s,
                s / 10.4 * (depth - 0.6),
                color="black",
            )
            ax.add_patch(circle)

    plt.ylim(-0.02, 0.1 * (depth - 1) + 0.02)
    plt.xlim(-0.02, 1.02)

    for layer_index in range(depth - 1):
        layer = layer_list[layer_index]
        w = layer.w.detach().cpu().numpy()
        if layer.sparse:
            w = w * torch.sigmoid(layer.w_gate).detach().cpu().numpy()
        in_dim = layer_size_list[layer_index]
        out_dim = layer_size_list[layer_index + 1]

        for output_neuron_index in range(out_dim):
            for input_neuron_index in range(in_dim):
                _w = w[input_neuron_index, output_neuron_index]
                width = np.max(np.abs([_w]))
                if _w > 0:
                    color = "red"
                else:
                    color = "blue"
                plt.plot(
                    [
                        1 / (2 * out_dim) + output_neuron_index / out_dim,
                        1 / (2 * in_dim) + input_neuron_index / in_dim,
                    ],
                    [0.1 * (depth - 2 - layer_index), 0.1 * (depth - 1 - layer_index)],
                    lw=0.5 * width,
                    color=color,
                )

    ax.axis("off")
    plt.show()




def plot_sns(model: Network) -> None:
    layer_size_list = model.layer_size_list
    layer_list = model.layer_list
    depth = model.depth + 1

    _, ax = plt.subplots(figsize=(3, 3))
    s = 1 / (2 * max(layer_size_list))

    for layer_index in range(depth):
        neuron_num = layer_size_list[layer_index]
        for neuron_index in range(neuron_num):
            circle = Ellipse(
                (
                    1 / (2 * neuron_num) + neuron_index / neuron_num,
                    0.1 * (depth - 1 - layer_index),
                ),
                s,
                s / 10.4 * (depth - 0.6),
                color="black",
            )
            ax.add_patch(circle)

    plt.ylim(-0.02, 0.1 * (depth - 1) + 0.02)
    plt.xlim(-0.02, 1.02)

    for layer_index in range(depth - 1):
        layer = layer_list[layer_index]
        w = layer.w.detach().cpu().numpy()
        v = layer.v.detach().cpu().numpy()
        if layer.sparse:
            w = w * torch.sigmoid(layer.w_gate).detach().cpu().numpy()
            v = w * torch.sigmoid(layer.v_gate).detach().cpu().numpy()
        in_dim = layer_size_list[layer_index]
        out_dim = layer_size_list[layer_index + 1]

        for output_neuron_index in range(out_dim):
            for input_neuron_index in range(in_dim):
                _w = w[input_neuron_index, output_neuron_index]
                _v = v[input_neuron_index, output_neuron_index]
                width = np.min([np.max(np.abs([_w, _v])), 1.5])
                if np.abs(_v) > np.abs(_w):
                    color = "green"
                else:
                    if _w > 0:
                        color = "red"
                    else:
                        color = "blue"
                plt.plot(
                    [
                        1 / (2 * out_dim) + output_neuron_index / out_dim,
                        1 / (2 * in_dim) + input_neuron_index / in_dim,
                    ],
                    [0.1 * (depth - 2 - layer_index), 0.1 * (depth - 1 - layer_index)],
                    lw=0.5 * width,
                    color=color,
                )

    ax.axis("off")
    plt.show()
