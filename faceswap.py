#!/usr/bin/env python3
import argparse
from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from scripts.convert import ConvertImage

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers()
extract = ExtractTrainingData(subparser, "extract", "Extract the faces from a pictures.")
convert = ConvertImage(subparser, "convert", "Convert a source image to a new one with the face swapped.")
train = TrainingProcessor(subparser, "train", "This command trains the model for the two faces A and B.")
arguments = parser.parse_args()
arguments.func(arguments)