"""
Converts between visual object detection dataset formats. See `converter.py` for more info.

To add support for additional data formats, define a module with an `converter.Ingestor` and/or
`converter.Egestor` implementation and add them to the `INGESTORS` and `EGESTORS` dicts below.
"""

import argparse
import logging

import umautobots_converter.converter as converter
import umautobots_converter.kitti as kitti
import umautobots_converter.kitti_tracking as kitti_tracking
import umautobots_converter.udacity as udacity
import umautobots_converter.voc as voc

import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# TODO: add split information to the ingestor
INGESTORS = {
    'kitti': kitti.KITTIIngestor(),
    'kitti-tracking': kitti_tracking.KITTITrackingIngestor(),
    'voc': voc.VOCIngestor(),
    'udacity-crowdai': udacity.UdacityCrowdAIIngestor(),
    'udacity-autti': udacity.UdacityAuttiIngestor()
}

EGESTORS = {
    'voc': voc.VOCEgestor(),
    'kitti': kitti.KITTIEgestor()
}


def main(*, from_path, from_key, to_path, to_key, select_only_known_labels=False, filter_images_without_labels=False, split=None):
    # JasonZheng16 might also have to do this with VOC Egestor
    ingestor = kitti.KITTIEgestor()
    egestor = voc.VOCEgestor()
    if from_key == 'kitti' and split:
        ingestor = kitti.KITTIIngestor(split=split)
    if to_key == 'voc' and split:
        egestor = voc.VOCEgestor(split=split)
    success, msg = converter.convert(from_path=from_path, ingestor=ingestor,
                                     to_path=to_path, egestor=egestor,
                                     select_only_known_labels=select_only_known_labels,
                                     filter_images_without_labels=filter_images_without_labels)
    if not success:
        print(f"Failed to convert from {from_key} to {to_key}: {msg}")
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description='Convert visual object datasets.')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--from',
                          dest='from_key',
                          required=True,
                          help=f'Format to convert from: one of {", ".join(INGESTORS.keys())}', type=str)
    required.add_argument('--from-path', dest='from_path',
                          required=True,
                          help=f'Path to dataset you wish to convert.', type=str)
    required.add_argument('--to', dest='to_key', required=True,
                          help=f'Format to convert to: one of {", ".join(EGESTORS.keys())}',
                          type=str)
    required.add_argument(
        '--to-path',
        dest='to_path', required=True,
        help="Path to output directory for converted dataset.", type=str)
    optional.add_argument(
        '--select-only-known-labels',
        help="only include labels known to the destination dataset (e.g skip 'trafficlight' if VOC doesn't know about it)",
        required=False,
        action='store_true',
        default=False
    )
    optional.add_argument(
        '--filter-images-without-labels',
        help="skip images that don't have any (known) labels",
        required=False,
        action='store_true',
        default=False
    )

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(from_path=args.from_path, from_key=args.from_key,
                  to_path=args.to_path, to_key=args.to_key,
                  select_only_known_labels=args.select_only_known_labels,
                  filter_images_without_labels=args.filter_images_without_labels))
