from segment_anything_gui import run_gui
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img", default="test.jpg", metavar="FILE", help="path to input image file")
    parser.add_argument("--output-img", default="tag.jpg", metavar="FILE", help="path to output tagged image file")
    return parser

if __name__ == "__main__":
    args = argument_parser().parse_args()
    segmenter = run_gui(args.input_img)
    if segmenter:
        segmenter.save_annotation(args.output_img)
