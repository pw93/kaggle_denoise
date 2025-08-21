# -*- coding: utf-8 -*-





def detect_kaggle_runtime():
    import os
    return os.path.exists("/kaggle/input")
