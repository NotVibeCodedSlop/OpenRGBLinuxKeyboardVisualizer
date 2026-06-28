# OpenRGBLinuxKeyboardVisualizer
A visualizer for sound and key presses for Linux OpenRGB with beat detection.
Well I made this using [Google AI Studio](https://aistudio.google.com) and well it should work for every 8-LED TKL keyboard but I made it for my SteelSeries Apex 3 TKL and I don't have any other adjustable RGB keyboards (at all).
It might need root but who knows.
Uhh dependancies?
```
import subprocess
import numpy as np
import os
import fcntl
import threading
import time
import random
from collections import deque
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor
```
Figure it out.
In the volume_viz.py is some config junk, just tune it.
If you want add pull requests but don't expect a response in that week.
