# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Adapted from Marllib

from typing import Any, Dict

from marllib.marl.algos.core.VD.iql_vdn_qmix import JointQTrainer
from marllib.marl.algos.scripts.coma import restore_model
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.rllib.agents.trainer_template import default_execution_plan
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.utils import merge_dicts

from niql.algo import ParamSharingQLearningPolicy
from niql.algo.vdn_qmix import JointQPolicy
from niql.utils import add_evaluation_config


def run_joint_q(model: Any, exp: Dict, run: Dict, env: Dict,
                stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the IQL, VDN, and QMIX algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    model_name = "Joint_Q_Model"
    ModelCatalog.register_custom_model(model_name, model)

    _param = AlgVar(exp)

    algorithm = exp["algorithm"]
    episode_limit = env["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    mixer_dict = {
        "qmix": "qmix",
        "vdn": "vdn",
    }

    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
            "custom_model": model_name,
            "fcnet_activation": back_up_config["model_arch_args"]["fcnet_activation"],
            "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"]
        },
    }

    config.update(run)
    config["simple_optimizer"] = _param.get("simple_optimizer", False)

    JointQ_Config.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": episode_limit * train_batch_episode,
            "lr": lr if restore is None else 1e-10,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "mixer": mixer_dict.get(algorithm)
        })

    JointQ_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    JointQ_Config["optimizer"] = optimizer
    JointQ_Config["training_intensity"] = None
    JointQ_Config["gamma"] = _param.get("gamma", JointQ_Config["gamma"])
    JointQ_Config["callbacks"] = _param.get("callbacks", JointQ_Config["callbacks"])

    JQTrainer = JointQTrainer.with_updates(
        name=algorithm.upper(),
        default_policy=JointQPolicy,
        default_config=JointQ_Config,
    )

    if algorithm.lower() == "iql":
        JQTrainer = JQTrainer.with_updates(
            default_policy=ParamSharingQLearningPolicy,
            execution_plan=default_execution_plan,
        )

    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    running_name = '_'.join([algorithm, arch, map_name])
    model_path = restore_model(restore, exp)

    # Periodic evaluation of trained policy
    config = add_evaluation_config(config)

    results = tune.run(JQTrainer,
                       name=running_name,
                       checkpoint_at_end=exp['checkpoint_end'],
                       checkpoint_freq=exp['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"])

    return results
