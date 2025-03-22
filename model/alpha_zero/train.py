import random, pickle, time
import numpy as np
from collections import deque

import zip_array
from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
# from mcts_pure import MCTS_Pure

# if CONFIG["use_frame"] == "paddle":
#     from paddle_net import PolicyValueNet
# elif CONFIG["use_frame"] == "pytorch":
from pytorch_net import PolicyValueNet
# else:
#     print("暂不支持您选择的框架")

class TrainPipeline:
    def __init__(self, init_model=None):
        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG["play_out"]
        self.c_puct = CONFIG["c_puct"]
        self.learn_rate = 1e-3
        self.lr_multiplier = 1
        self.temp = 1.0
        self.batch_size = CONFIG["batch_size"]
        self.epochs = CONFIG["epochs"]
        self.kl_targ = CONFIG["kl_targ"]
        self.check_freq = 100
        self.game_batch_num = CONFIG["game_batch_num"]
        self.best_win_ratio = 0.0
        self.buffer_size = CONFIG["buffer_size"]
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print("已加载上次最终模型")
            except:
                print("模型路径不存在, 从零开始训练")
                self.policy_value_net = PolicyValueNet()
        else:
            print("模型路径不存在, 从零开始训练")
            self.policy_value_net = PolicyValueNet()
    
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype("float32")

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype("float32")

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype("float32")

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.9f},explained_var_new:{:.9f}".format(
            kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new
        ))
    
    def run(self):
        try:
            for i in range(self.game_batch_num):
                while True:
                    try:
                        with open(CONFIG["train_data_buffer_path"]) as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file["data_buffer"]
                            self.iters = data_file["iters"]
                            del data_file
                        print("已加载数据")
                        break
                    except:
                        time.sleep(30)
                print("step i {}: ".format(self.iters))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    if CONFIG["use_frame"] == "paddle":
                        self.policy_value_net.save_model(CONFIG["paddle_model_path"])
                    elif CONFIG["use_frame"] == "pytorch":
                        self.policy_value_net.save_model(CONFIG["pytorch_model_path"])
                    else:
                        print("不支持所选框架")
                time.sleep(CONFIG["train_update_interval"])
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    self.policy_value_net.save_model("models/current_policy_batch{}.pkl".format(i + 1))
        except KeyboardInterrupt:
            print("\r\nquit")

if CONFIG["use_frame"] == "paddle":
    training_pipeline = TrainPipeline(init_model="current_policy.model")
    training_pipeline.run()
elif CONFIG["use_frame"] == "pytorch":
    training_pipeline = TrainPipeline(init_model="current_policy.pkl")
    training_pipeline.run()
else:
    print("暂不支持您选择的框架")
    print("训练结束")

