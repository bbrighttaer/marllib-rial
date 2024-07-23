import copy

import torch
from ray.rllib.agents.qmix.qmix_policy import _drop_agent_dim
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.preprocessors import get_preprocessor


class DotDic(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


def notify_wrap(f, cb):
    def wrapped(*args, **kwargs):
        f(*args, **kwargs)
        cb(*args, **kwargs)

    return wrapped


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def get_group_rewards(n_agents, info_batch):
    group_rewards = np.array([
        info.get("_group_rewards", [0.0] * n_agents)
        for info in info_batch
    ])
    return group_rewards


def standardize(r):
    return (r - r.mean()) / (r.std() + 1e-5)


def tb_add_scalar(policy, label, value):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalar(policy.policy_id + "/" + label, value, policy.global_timestep)


def tb_add_histogram(policy, label, data):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_histogram(policy.policy_id + "/" + label, data.reshape(-1, ), policy.global_timestep)


def tb_add_scalars(policy, label, values_dict):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalars(
            policy.policy_id + "/" + label, {str(k): v for k, v in values_dict.items()}, policy.global_timestep
        )


def apply_scaling(vector):
    # Compute scaling factor
    scaling = len(vector) / vector.sum()

    # Scale the vector
    vector *= scaling

    # Normalise to [0, 1]
    vector /= (vector.max() + 1e-7)
    return vector


def shift_and_scale(x):
    # Find the minimum value
    x_min = x.min()

    # Shift the vector to make all elements non-negative
    x_shifted = (x - x_min) + 1e-4

    # Normalise to [0, 1]
    x_scaled = x_shifted / (x_shifted.max() + 1e-7)

    return x_scaled


def mac(model, obs, h, **kwargs):
    """Forward pass of the multi-agent controller.

    Args:
        model: TorchModelV2 class
        obs: Tensor of shape [B, n_agents, obs_size]
        h: List of tensors of shape [B, n_agents, h_size]

    Returns:
        q_vals: Tensor of shape [B, n_agents, n_actions]
        h: Tensor of shape [B, n_agents, h_size]
    """
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_agents_as_batches = {k: _drop_agent_dim(v) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    q_flat, h_flat = model(obs_agents_as_batches, h_flat, None, **kwargs)
    return q_flat.reshape(
        [B, n_agents, -1]), [s.reshape([B, n_agents, -1]) for s in h_flat]


def unroll_mac(model, obs_tensor, **kwargs):
    """Computes the estimated Q values for an entire trajectory batch"""
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    mac_h_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]

    # forward propagation through time
    for t in range(T):
        # get input data for this time step
        obs = obs_tensor[:, t]

        # build new args with values of current time step
        _kwargs = {
            k: kwargs[k][:, t].reshape(B * n_agents, -1) for k in kwargs.keys()
        }

        # forward propagation
        q, h = mac(model, obs, h, **_kwargs)
        mac_out.append(q)
        mac_h_out.extend(h)
    mac_out = torch.stack(mac_out, dim=1)  # Concat over time
    mac_h_out = torch.stack(mac_h_out, dim=1)

    return mac_out, mac_h_out


def unroll_mac_squeeze_wrapper(model_outputs):
    pred, hs = model_outputs
    return pred.squeeze(2), hs.squeeze(2)


def soft_update(target_net, source_net, tau):
    """
    Soft update the parameters of the target network with those of the source network.

    Args:
    - target_net: Target network.
    - source_net: Source network.
    - tau: Soft update parameter (0 < tau <= 1).

    Returns:
    - target_net: Updated target network.
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    return target_net


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def add_evaluation_config(config: dict) -> dict:
    config = dict(config)
    config.update({
        "evaluation_interval": 1,
        "evaluation_num_episodes": 20,
        "evaluation_num_workers": 1,
        # "evaluation_unit": "timesteps", # not supported in ray 1.8.0
    })
    return config
