# pylint: disable=missing-docstring,wrong-import-position

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, "slim")

from gpkg.slim.datasets import prepare_custom

from guild import util # pylint: disable=import-error

def main():
    args, other_args = _parse_args()
    if args.dataset == "dogs-vs-cats":
        working = _prepare_cats_and_dogs(args)
    else:
        raise AssertionError(args.dataset)
    sys.argv = [sys.argv[0], "--images-dir", working] + other_args
    prepare_custom.main()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("dogs-vs-cats",))
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--working-dir")
    return parser.parse_known_args()

def _prepare_cats_and_dogs(args):
    working = args.working_dir or "images"
    util.ensure_dir(working)
    for src_name in os.listdir(args.images_dir):
        cat, dest_name = src_name.split(".", 1)
        cat_dir = os.path.join(working, cat)
        dest = os.path.join(cat_dir, dest_name)
        if not os.path.exists(dest):
            src = os.path.relpath(
                os.path.join(args.images_dir, src_name),
                cat_dir)
            util.ensure_dir(cat_dir)
            os.symlink(src, dest)
    return working

if __name__ == "__main__":
    main()
