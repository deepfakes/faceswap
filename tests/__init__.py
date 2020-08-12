#!/usr/bin/env python3
""" Use custom Importer for importing Keras for tests """
import sys
from lib.utils import KerasFinder


sys.meta_path.insert(0, KerasFinder())
