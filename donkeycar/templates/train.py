#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|pilotnet|mlp|categorical|memory|rnn|cnn_lstm|confidence|vit|world_model|diffusion_policy|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]
    [--mask-glare | --no-mask-glare]
    [--style-transfer | --no-style-transfer]
    [--style-transfer-preset=<preset>]
    [--style-transfer-blend=<blend>]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    mask_glare = True if args['--mask-glare'] else False if args['--no-mask-glare'] else None
    style_transfer = True if args['--style-transfer'] else False if args['--no-style-transfer'] else None
    style_transfer_preset = args['--style-transfer-preset']
    style_transfer_blend = float(args['--style-transfer-blend']) if args['--style-transfer-blend'] is not None else None
    train(cfg, tubs, model, model_type, comment,
          mask_glare=mask_glare,
          style_transfer=style_transfer,
          style_transfer_preset=style_transfer_preset,
          style_transfer_blend=style_transfer_blend)


if __name__ == "__main__":
    main()
