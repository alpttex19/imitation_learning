import time
import numpy as np

from pyjoycon import JoyCon, get_R_id

from ..handle import Handle


class PickBoxJoyconHandle(Handle):
    def __init__(self):
        super().__init__()

        self._joycon_R = JoyCon(*get_R_id())

