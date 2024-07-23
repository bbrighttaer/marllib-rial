import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from marllib import marl

from niql import envs, scripts, seed
from niql.models import *  # noqa

logger = logging.getLogger(__name__)

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-a', '--algo',
        type=str,
        default='vdn',
        choices=['vdn', 'qmix', 'iql'],
        help='Select which CTDE algorithm to run.',
    )

    args = parser.parse_args()

    mode = args.exec_mode

    # get env
    env, exp_config = envs.get_active_env()

    # register execution script
    marl.algos.register_algo(
        algo_name=args.algo,
        style="VD",
        script=scripts.run_joint_q,
    )

    # initialise algorithm with hyperparameters
    algo = getattr(marl.algos, args.algo)
    algo.algo_parameters = exp_config['algo_parameters']

    # build model
    model_config = exp_config['model_preference']
    # model_config.update({'core_arch': args.model_arch})
    model = marl.build_model(env, algo, model_preference=exp_config['model_preference'])
    if model_config.get('model'):
        model = (eval(model_config['model']), model[1])

    gpu_count = torch.cuda.device_count()

    # start training
    algo.fit(
        env,
        model,
        stop=exp_config['stop_condition'],
        local_mode=gpu_count == 0,
        num_gpus=gpu_count,
        num_workers=5,
        share_policy='all',
        checkpoint_freq=100,
    )
