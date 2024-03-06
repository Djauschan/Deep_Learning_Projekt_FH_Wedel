import torch
import glob
from tkinter import Place
# from seeq import spy
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import date as dt, datetime
import yaml
from yaml.loader import SafeLoader
import time


class ConfigService:
    def __init__(self):
        pass

    def loadModelConfig(self, path):
        # opening a file
        parameter = {}
        with open(path, 'r') as stream:
            try:
                # Converts yaml document to python object
                parameter = yaml.load(stream, Loader=SafeLoader)
            except yaml.YAMLError as e:
                print('CONFOG NOT FOUND')
                print(e)
                # todo feherlbehandung einbauen

        return parameter
