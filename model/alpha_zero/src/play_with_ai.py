from config import CONFIG
from game import move_action2move_id, Board
from mcts import MCTSPlayer, Game
from pytorch_net import PolicyValueNet

class Human1:
    def get_action(self, board: Board):
        move_id = move_action2move_id[input("请输入 move_action: ")]
        return move_id
    
    def set_player_ind(self, p):
        self.player_id = p

policy_value_net = PolicyValueNet(model_file=CONFIG["pytorch_model_path"])
mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                         c_puct=5, n_playout=1000, is_selfplay=0)
human = Human1()
game = Game(board=Board())
game.start_play(mcts_player, human, start_player=1, is_shown=1)
