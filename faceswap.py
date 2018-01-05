#!/usr/bin/env python3
from lib.utils import FullHelpArgumentParser

from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from scripts.convert import ConvertImage

if __name__ == "__main__":
    parser = FullHelpArgumentParser()
    subparser = parser.add_subparsers()
    extract = ExtractTrainingData(
        subparser, "extract", "Extract the faces from a pictures.")
    train = TrainingProcessor(
        subparser, "train", "This command trains the model for the two faces A and B.")
    convert = ConvertImage(
        subparser, "convert", "Convert a source image to a new one with the face swapped.")
    arguments = parser.parse_args()
    arguments.func(arguments)
