import numpy as np
import copy
from config import CONFIG

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

    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择 Q+U 最大的节点"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算并返回此节点的 Q+U 值"""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """更新本节点"""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归地更新本节点的所有直系节点"""
        if self._parent:
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
        :param policy_value_fn: 输入盘面, 返回落子概率和盘面评估得分
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        进行一次搜索
        :param state: 盘面
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        
        action_probs, leaf_value = self._policy(self._c_puct)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player_id() else -1.0
                )
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        按顺序运行所有搜索并返回可用的动作及相应的概率
        :param state: 盘面
        :param temp: 介于(0, 1]之间的温度参数
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy()
            self._playout(state_copy)
        
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        
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
        self.mcts.update_with_move(-1)
    
    def get_action(self, board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(2086)
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move






