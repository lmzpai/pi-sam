import argparse
import tempfile
from functools import partial
from pathlib import Path
import sys
sys.path.extend(['.', '..'])
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from segment_anything import sam_model_registry, build_mask_decoder

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def inference():
    ##################
    # fix this
    model = build_model()
    ##################

    model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    img_shape = (3, 1024, 1024)
    device=next(model.parameters()).device
    img = torch.randn(img_shape, device=device)

    result = {}
    avg_flops = []
    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=img,
        show_table=True,
        show_arch=True)
    avg_flops.append(outputs['flops'])
    params = outputs['params']
    result['compute_type'] = 'dataloader: load a picture from the dataset'

    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params

    print(outputs['out_table'])

    return result


def main():
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference()


if __name__ == '__main__':
    main()