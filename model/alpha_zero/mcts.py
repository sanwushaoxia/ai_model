import copy, time
import numpy as np
from config import CONFIG
from game import print_board, state_list2state_array, Board, move_id2move_action

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        """
        :param parent: 当前节点的父节点
        :param prior_p: 当前节点被选择的先验概率
        """
        self._parent = parent
        self._children = {} # key: action; value: TreeNode
        self._n_visits = 0  # 当前节点的访问次数
        self._Q = 0         # 当前节点的平均价值
        self._u = 0         # 当前节点的置信上限
        self._P = prior_p

    def expand(self, moves_id2probs):
        """
        扩展子节点
        moves_id2probs: moves_id 到 probs 的映射
        """
        for action, prob in moves_id2probs:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def get_value(self, c_puct):
        """
        计算并返回此节点的 Q+U 值
        c_puct: 常系数
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def select(self, c_puct):
        """
        在子节点中选择 Q+U 最大的节点
        """
        # act_node[1]: TreeNode
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """更新本节点"""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归地更新本节点的所有直系节点"""
        if self._parent:
            # 在生成对抗网络中, 更新该节点和更新该节点的父节点的作用相反
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """本节点是否是叶节点"""
        return self._children == {}

    def is_root(self):
        """本节点是否是根节点"""
        return self._parent is None

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        :param policy_value_fn: 输入盘面, 返回走子到落子概率的映射以及盘面评估得分
        :param c_puct: 常系数
        """
        self._root = TreeNode(None, 1.0) # 根节点
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout # 探索次数

    def _playout(self, state: Board):
        """
        进行一次搜索
        :param state: 盘面
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            moves_id, node = node.select(self._c_puct)
            state.do_move(moves_id)
        
        moves_id2probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(moves_id2probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player_id() else -1.0
                )
        # 必须使用 -leaf_value 进行递归更新, 原因在于 state.do_move 后,
        # state.get_current_player_id 会更新, 即若 winner == current_player_id,
        # 则表示当前走子导致输棋, 当前节点对当前玩家的价值降低.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state: Board, temp=1e-3):
        """
        按顺序运行所有搜索并返回可行的动作及相应的概率
        :param state: 盘面
        :param temp: 介于(0, 1]之间的温度参数
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # 依据通过多次探索得到的各节点访问次数生成概率
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        根据 last_move, 生成子树或新树
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

def move2idx(move, acts):
    for i in range(len(acts)):
        if acts[i] == move:
            return i
    return None

class MCTSPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"
    
    def set_player_ind(self, p):
        self.player_id = p
    
    def reset_player(self):
        """生成新的树"""
        self.mcts.update_with_move(-1)
    
    def get_action(self, board: Board, temp=1e-3, return_prob=0):
        # 2086 表示所有合法的走子数量
        moves_id2probs = np.zeros(2086)
        # acts 表示可行的动作, probs 表示相应的概率
        acts, probs = self.mcts.get_move_probs(board, temp)
        # 依据当前搜索树, 给出所有可能的走子概率
        moves_id2probs[list(acts)] = probs
        if self._is_selfplay:
            # 使用 Dirichlet Noise, 以增强探索
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            # 非自我博弈时, 由于双方交替走子, 所以本次使用的搜索树和下次使用的搜索树不易共用
            move = np.random.choice(acts, p=probs)
            print("idx: ", move2idx(move, acts))
            print("move_id: ", move)
            print("move_action: ", move_id2move_action[move])
            print("acts: ", acts)
            print("probs: ", probs)
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, moves_id2probs
        else:
            return move

class Game(object):
    def __init__(self, board: Board):
        self.board = board
    
    def graphic(self, board: Board, player1_id, player2_id):
        """
        param: board     : 需要图形化最新棋盘的棋盘类
        param: player1_id: 玩家 1 使用的颜色
        param: player2_id: 玩家 2 使用的颜色
        """
        print("player1 take:", player1_id)
        print("player2 take:", player2_id)
        print_board(state_list2state_array(board.state_deque[-1]))

    # 等实现 MCTSPlayer 类后再看
    def start_play(self, player1: MCTSPlayer, player2: MCTSPlayer, start_player=1, is_shown=1):
        """
        :param player1     : 使用 ID=1 的玩家
        :param player2     : 使用 ID=2 的玩家
        :param start_player: 先手玩家 ID
        :param is_shown    : 1 表示每次移动后打印最新的棋盘
        返回赢家的 ID
        """
        if start_player not in (1, 2):
            raise Exception("start_player must be either 1 or 2")
        self.board.init_board(start_player)
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        player_id2player = {1: player1, 2: player2}
        if is_shown:
            self.graphic(self.board, player1.player_id, player2.player_id)
        
        while True:
            current_player_id = self.board.get_current_player_id()
            player_in_turn = player_id2player[current_player_id]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player_id, player2.player_id)
            end, winner = self.board.game_end()
            if end:
                print("Game end. Winner is ", player_id2player[winner])
                return winner

    def start_self_play(self, player: MCTSPlayer, is_shown=False, temp=1e-3):
        """
        开始自我博弈, 并收集数据
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, moves_id2probs = player.get_action(self.board, temp=temp, return_prob=1)
                print("走一步要花: ", time.time() - start_time)
            else:
                move, moves_id2probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 将当前棋盘添加至列表中
            states.append(self.board.current_state())
            # 将当前棋盘可行的走子及对应概率添加至列表中
            mcts_probs.append(moves_id2probs)
            # 将当前走子的玩家 ID 添加至列表中
            current_players.append(self.board.current_player_id)
            # 走子
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                winner_z = np.ones(len(current_players))
                winner_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    print("Game end. Winner is ", winner)
                return winner, zip(states, mcts_probs, winner_z)
