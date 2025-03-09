import copy
import numpy as np
from config import CONFIG
from game import Board

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
        
        moves_id2probs, leaf_value = self._policy(self._c_puct)
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
        # 源代码使用 -leaf_value 进行递归更新, 当 winner == state.get_current_player_id()
        # 理应对该节点有正向作用, 使用 1.0 而非 -1.0 进行更新
        node.update_recursive(leaf_value)

    def get_move_probs(self, state: Board, temp=1e-3):
        """
        按顺序运行所有搜索并返回可用的动作及相应的概率
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

class MCTSPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"
    
    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        """生成新的树"""
        self.mcts.update_with_move(-1)
    
    def get_action(self, board: Board, temp=1e-3, return_prob=0):
        # 2086 表示所有合法的走子数量
        moves_id2probs = np.zeros(2086)
        acts, probs = self.mcts.get_move_probs(board, temp)
        # 依据当前搜索树, 给出所有可能的走子概率
        moves_id2probs[list(acts)] = probs
        if self._is_selfplay:
            # 使用 Dirichlet Noise 自我博弈
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            # 非自我博弈时, 由于双方交替走子, 所以本次使用的搜索树和下次使用的搜索树不易共用
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, moves_id2probs
        else:
            return move
