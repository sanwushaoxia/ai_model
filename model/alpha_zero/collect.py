
from collections import deque
import copy, os, pickle, time

from game import Board, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer, Game
import zip_array
from config import CONFIG

if CONFIG["use_frame"] == "paddle":
    # from paddle_net import PolicyValueNet
    pass
elif CONFIG["use_frame"] == "pytorch":
    from pytorch_net import PolicyValueNet
else:
    print("暂不支持您选择的框架")

class CollectPipeline:
    def __init__(self, init_model=None):
        """
        init_model: 当前未被使用, 后续开发该参数使用场景
        """
        self.board = Board()
        self.game = Game(self.board)
        # 参数
        self.temp = 1
        self.n_playout = CONFIG["play_out"]
        self.c_puct = CONFIG["c_puct"]
        self.buffer_size = CONFIG["buffer_size"]
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.init_model = init_model
    
    def load_model(self):        
        try:
            self.policy_value_net = PolicyValueNet(model_file=self.init_model)
            print("已加载最新模型")
        except:
            self.policy_value_net = PolicyValueNet()
            print("已加载初始模型")
        
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
    
    def get_equi_data(self, play_data):
        "左右对称变换以扩充数据"
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            # 第一维是特征, 将其移至最后一维
            state_flip = state.transpose([1, 2, 0]) # 10 * 9 * 9
            state = state.transpose([1, 2, 0]) # 10 * 9 * 9
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            # 将最后一维还原至第一维
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我博弈数据"""
        for i in range(n_games):
            self.load_model()
            # play_data 的格式为 zip(states, mcts_probs, winner_z)
            _, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)
            self.play_data_len = len(play_data)
            # 扩充数据
            play_data = self.get_equi_data(play_data)
            if os.path.exists(CONFIG["train_data_buffer_path"]):
                while True:
                    try:
                        with open(CONFIG["train_data_buffer_path"], "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = deque(maxlen=self.buffer_size)
                            # 将 data_file 中的 data_buffer 添加至 deque 中
                            self.data_buffer.extend(data_file["data_buffer"])
                            self.iters = data_file["iters"]
                            del data_file
                            self.iters += 1
                            # 将刚收集的 play_data 也添加至 deque 中
                            self.data_buffer.extend(play_data)
                        print("收集数据完成")
                        break
                    except:
                        print("收集数据失败")
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.iters += 1
            # 将最新的数据保存至文件中
            data_dict = {"data_buffer": self.data_buffer, "iters": self.iters}
            with open(CONFIG["train_data_buffer_path"], "wb") as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters
    
    def run(self):
        "收集数据"
        try:
            while True:
                iters = self.collect_selfplay_data()
                print("batch i: {}, play_data_len: {}".format(iters, self.play_data_len))
        except KeyboardInterrupt:
            print("\r\nquit")

if CONFIG['use_frame'] == 'pytorch':
    collecting_pipeline = CollectPipeline()
    collecting_pipeline.run()
else:
    print("暂不支持您选择的框架")
    print("训练结束")

# if __name__ == "__main__":
#     l1 = [1, 2, 3]
#     l2 = [4, 5, 6]
#     l3 = [7, 8, 9]
#     print(list(zip(l1, l2, l3)))
