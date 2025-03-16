import random
from collections import deque
import copy, os, pickle, time
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
import zip_array
from config import CONFIG

# if CONFIG["use_redis"]:
#     import my_redis, redis

if CONFIG["use_frame"] == "paddle":
    from paddle_net import PolicyValueNet
elif CONFIG["use_frame"] == "pytorch":
    from pytorch_net import PolicyValueNet
else:
    print("暂不支持您选择的框架")


