import copy, time
from collections import deque
import numpy as np
from config import CONFIG

# 棋盘初始状态, 使用时需要对其进行深拷贝
state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]

# 长度为4, 用于记录之前4次棋盘的状态, 避免双方循环相同的走法
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))

# 字典: 对棋子进行 onehot 编码, 将字符串映射为数组, 红色棋子用1表示, 黑色棋子用-1表示
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    红马=np.array([0, 1, 0, 0, 0, 0, 0]), 黑马=np.array([0, -1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    红士=np.array([0, 0, 0, 1, 0, 0, 0]), 黑士=np.array([0, 0, 0, -1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    红炮=np.array([0, 0, 0, 0, 0, 1, 0]), 黑炮=np.array([0, 0, 0, 0, 0, -1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]))

# 获取 onehot 编码对应的棋子, 为 string2array 的逆映射
def array2string(array):
    """
    :param array: numpy数组
    """
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]

# 返回对当前棋盘 state_list 进行 move 操作后的棋盘, 既适用于吃子, 也适用于不吃子
def change_state(state_list, move: str):
    """
    :param state_list: 棋盘, 二维list, 例如 state_list_init
    :param move:       字符串, 第一个字符表示源y轴坐标, 第二个字符表示源x轴坐标, 第三个字符表示目的y坐标, 第四个字符表示目的x坐标,
                       对于棋盘, 左上角为原点, 向下为y轴正方向, 向右为x轴正方向
    """
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list

# 对使用 onehot 编码的棋盘进行解码
def print_board(_state_array):
    """
    :param _state_array: numpy数组
    """
    board_line = []
    for i in range(10): # y轴长度
        for j in range(9): # x轴长度
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()

# 对棋盘进行 onehot 编码
def state_list2state_array(state_list):
    """
    :param state_list: 二维list
    """
    _state_array = np.zeros([10, 9, 7]) # H * W * C
    for i in range(10): # y轴长度
        for j in range(9): # x轴长度
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array

# 检查棋子是否在棋盘上
def check_bounds(toY, toX, chess: str = None):
    if chess != None:
        if chess == '黑象' and toY < 5:
            return False
        elif chess == '红象' and toY > 4:
            return False
        elif (chess == '黑士' or chess == '黑帅') and (toY >= 7 and 3 <= toX <= 5) == False:
            return False
        elif (chess == '红士' or chess == '红帅') and (toY <= 2 and 3 <= toX <= 5) == False:
            return False
    if toY in range(10) and toX in range(9):
        return True
    return False
    

# 获取所有合法移动集合, 并建立 id 与移动的双向映射关系
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2806', '0628', '2846', '4628', '0624', '2406', '4624', '2446',
                     '7092', '9270', '7052', '5270', '9274', '7492', '5274', '7452',
                     '7896', '9678', '7856', '5678', '9674', '7496', '5674', '7456']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (2, 1), (-1, -2), (1, 2), (-2, 1), (2, -1), (-1, 2), (1, -2)]] # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and check_bounds(l2, n2):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1
    
    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1
    
    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1
    
    return _move_id2move_action, _move_action2move_id

# 获取与指定移动左右对称的移动
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str

# 检查棋子移动是否会遇到阻碍, 返回 True 表示没有阻碍, 返回 False 表示有阻碍
def check_obstruct(piece, current_player_color):
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True

move_id2move_action, move_action2move_id = get_all_legal_moves()

def add_valid_move(m, moves: list, state_list, old_state_list):
    # 若非长捉, 则该移动为有效移动, 添加m至moves列表中
    if change_state(state_list, m) != old_state_list:
        moves.append(m)

def add_car_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '车':
        return
    if current_player_color == state_list[y][x][0]: # 车的合理走法
        toY = y
        # 车往左走
        for toX in range(x - 1, -1, -1):
            m = str(y) + str(x) + str(toY) + str(toX)
            if state_list[toY][toX] != '一一':
                if current_player_color not in state_list[toY][toX]:
                    add_valid_move(m, moves, state_list, old_state_list)
                break
            add_valid_move(m, moves, state_list, old_state_list)
        # 车往右走
        for toX in range(x + 1, 9):
            m = str(y) + str(x) + str(toY) + str(toX)
            if state_list[toY][toX] != '一一':
                if current_player_color not in state_list[toY][toX]:
                    add_valid_move(m, moves, state_list, old_state_list)
                break
            add_valid_move(m, moves, state_list, old_state_list)
        
        toX = x
        # 车往上走
        for toY in range(y - 1, -1, -1):
            m = str(y) + str(x) + str(toY) + str(toX)
            if state_list[toY][toX] != '一一':
                if current_player_color not in state_list[toY][toX]:
                    add_valid_move(m, moves, state_list, old_state_list)
                break
            add_valid_move(m, moves, state_list, old_state_list)
        # 车往下走
        for toY in range(y + 1, 10):
            m = str(y) + str(x) + str(toY) + str(toX)
            if state_list[toY][toX] != '一一':
                if current_player_color not in state_list[toY][toX]:
                    add_valid_move(m, moves, state_list, old_state_list)
                break
            add_valid_move(m, moves, state_list, old_state_list)

def add_horse_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '马':
        return
    if state_list[y][x][0] == current_player_color:
        for i in range(-1, 3, 2):
            for j in range(-1, 3, 2):
                toY = y + 2 * i
                toX = x + 1 * j
                if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color) and state_list[toY - i][x] == '一一':
                    m = str(y) + str(x) + str(toY) + str(toX)
                    add_valid_move(m, moves, state_list, old_state_list)
                toY = y + 1 * i
                toX = x + 2 * j
                if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color) and state_list[y][toX - j] == '一一':
                    m = str(y) + str(x) + str(toY) + str(toX)
                    add_valid_move(m, moves, state_list, old_state_list)

def add_elephant_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '象':
        return
    if state_list[y][x][0] == current_player_color:
        for i in range(-2, 3, 4):
            toY = y + i
            toX = x + i
            if check_bounds(toY, toX, state_list[y][x]) and check_obstruct(state_list[toY][toX], current_player_color) and state_list[y + i // 2][x + i // 2] == '一一':
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
            toY = y + i
            toX = x - i
            if check_bounds(toY, toX, state_list[y][x]) and check_obstruct(state_list[toY][toX], current_player_color) and state_list[y + i // 2][x - i // 2] == '一一':
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)

def add_shi_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '士':
        return
    if state_list[y][x][0] == current_player_color:
        for i in range(-1, 3, 2):
            toY = y + i
            toX = x + i
            if check_bounds(toY, toX, state_list[y][x]) and check_obstruct(state_list[toY][toX], current_player_color):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
            toY = y + i
            toX = x - i
            if check_bounds(toY, toX, state_list[y][x]) and check_obstruct(state_list[toY][toX], current_player_color):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)

def add_shuai_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][0] == current_player_color:
        for i in range(2):
            for sign in range(-1, 2, 2):
                toY = y + i * sign
                toX = x + (1 - i) * sign

                if check_bounds(toY, toX, state_list[y][x]) and check_obstruct(state_list[toY][toX], current_player_color):
                    m = str(y) + str(x) + str(toY) + str(toX)
                    if change_state(state_list, m) != old_state_list:
                        moves.append(m)

def add_pao_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '炮':
        return
    if state_list[y][x][0] == current_player_color:
        toY = y
        hits = False
        for toX in range(x - 1, -1, -1):
            m = str(y) + str(x) + str(toY) + str(toX)
            if hits is False:
                if state_list[toY][toX] != '一一':
                    hits = True
                else:
                    if change_state(state_list, m) != old_state_list:
                        moves.append(m)
            else:
                if state_list[toY][toX] != '一一':
                    if current_player_color not in state_list[toY][toX]:
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    break
        hits = False
        for toX in range(x + 1, 9):
            m = str(y) + str(x) + str(toY) + str(toX)
            if hits is False:
                if state_list[toY][toX] != '一一':
                    hits = True
                else:
                    if change_state(state_list, m) != old_state_list:
                        moves.append(m)
            else:
                if state_list[toY][toX] != '一一':
                    if current_player_color not in state_list[toY][toX]:
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    break
        
        toX = x
        hits = False
        for toY in range(y - 1, -1, -1):
            m = str(y) + str(x) + str(toY) + str(toX)
            if hits is False:
                if state_list[toY][toX] != '一一':
                    hits = True
                else:
                    if change_state(state_list, m) != old_state_list:
                        moves.append(m)
            else:
                if state_list[toY][toX] != '一一':
                    if current_player_color not in state_list[toY][toX]:
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    break
        hits = False
        for toY in range(y + 1, 10):
            m = str(y) + str(x) + str(toY) + str(toX)
            if hits is False:
                if state_list[toY][toX] != '一一':
                    hits = True
                else:
                    if change_state(state_list, m) != old_state_list:
                        moves.append(m)
            else:
                if state_list[toY][toX] != '一一':
                    if current_player_color not in state_list[toY][toX]:
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    break

def add_bing_moves(state_list, old_state_list, current_player_color, y, x, moves):
    if state_list[y][x][1] != '兵' or state_list[y][x][0] != current_player_color:
        return
    if current_player_color == '黑': # 黑兵的合理走法
        toY = y - 1
        toX = x
        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑'):
            m = str(y) + str(x) + str(toY) + str(toX)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        # 兵过河
        if y < 5:
            toY = y
            toX = x + 1
            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑'):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
            toX = x - 1
            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑'):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
    elif current_player_color == '红': # 红兵的合理走法
        toY = y + 1
        toX = x
        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红'):
            m = str(y) + str(x) + str(toY) + str(toX)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        # 兵过河
        if y > 4:
            toY = y
            toX = x + 1
            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红'):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
            toX = x - 1
            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红'):
                m = str(y) + str(x) + str(toY) + str(toX)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)

def get_legal_moves(state_deque, current_player_color):
    state_list = state_deque[-1] # 最新棋盘
    old_state_list = state_deque[-4]
    moves = [] # 记录棋盘上所有棋子的合理走法
    face_to_face = False # 红帅与黑帅在没有其它棋子的同一条线上

    black_x = None # 记录黑帅的位置
    black_y = None # 记录黑帅的位置
    red_x = None # 记录红帅的位置
    red_y = None # 记录红帅的位置

    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '一一':
                pass
            else:
                add_car_moves(state_list, old_state_list, current_player_color, y, x, moves)

                add_horse_moves(state_list, old_state_list, current_player_color, y, x, moves)

                add_elephant_moves(state_list, old_state_list, current_player_color, y, x, moves)

                add_shi_moves(state_list, old_state_list, current_player_color, y, x, moves)

                add_pao_moves(state_list, old_state_list, current_player_color, y, x, moves)

                add_bing_moves(state_list, old_state_list, current_player_color, y, x, moves)

                if state_list[y][x] == '黑帅': # 黑帅的合理走法
                    black_x = x
                    black_y = y
                    add_shuai_moves(state_list, old_state_list, current_player_color, y, x, moves)

                elif state_list[y][x] == '红帅': # 红帅的合理走法
                    red_x = x
                    red_y = y
                    add_shuai_moves(state_list, old_state_list, current_player_color, y, x, moves)

    if red_x is not None and black_x is not None and red_x == black_x:
        face_to_face = True
        for i in range(red_y + 1, black_y, 1):
            if state_list[i][red_x] != '一一':
                face_to_face = False
    
    if face_to_face is True:
        if current_player_color == '黑':
            m = str(black_y) + str(black_x) + str(red_y) + str(red_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        else:
            m = str(red_y) + str(red_x) + str(black_y) + str(black_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
    
    moves_id = []
    for move in moves:
        moves_id.append(move_action2move_id[move])
    return moves_id

class Board(object):
    def __init__(self):
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)
    
    def init_board(self, start_player=1):
        """
        :param start_player: 先手玩家 ID
        """
        self.start_player = start_player
        if start_player == 1:
            self.id2color = {1: "红", 2: "黑"}
            self.color2id = {"红": 1, "黑": 2}
            self.backhand_player = 2
        elif start_player == 2:
            self.id2color = {2: "红", 1: "黑"}
            self.color2id = {"红": 2, "黑": 1}
            self.backhand_player = 1
        # 当前手玩家, 即先手玩家
        self.current_player_color = "红"
        self.current_player_id    = self.color2id["红"]
        # 初始化棋盘
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        # 记录对方最后 move_id
        self.last_move = -1
        # 记录游戏中2次吃子间隔的回合数
        self.kill_action = 0
        # 记录游戏是否开始
        self.game_start = False
        # 游戏动作计数
        self.action_count = 0
        # 记录胜者的玩家id
        self.winner = None

    @property
    def availables(self):
        """获取当前盘面所有合法走法"""
        return get_legal_moves(self.state_deque, self.current_player_color)

    def current_state(self):
        """
        使用 9 个特征表示棋盘
        第 0-6 个特征表示不同棋子, 1 代表红, -1 代表黑
        第 7 个特征表示对方最后一步落子位置, 走子之前为 -1, 走子之后为 1, 其余为 0
        第 8 个特征表示本方是不是先手, 如果是先手, 那么全为 1, 否则全为 0
        """
        _current_state = np.zeros([9, 10, 9])
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1]) # [7, 10, 9]
        if self.game_start:
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0
        return _current_state
    
    def do_move(self, move_id):
        self.game_start = True
        self.action_count += 1
        move_action = move_id2move_action[move_id]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])

        if state_list[end_y][end_x] != "一一":
            self.kill_action = 0
            if self.current_player_color == "黑" and state_list[end_y][end_x] == "红帅":
                self.winner = self.color2id["黑"]
            elif self.current_player_color == "红" and state_list[end_y][end_x] == "黑帅":
                self.winner = self.color2id["红"]
        else:
            self.kill_action += 1
        
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = "一一"
        self.current_player_color = "黑" if self.current_player_color == "红" else "红"
        self.current_player_id = 1 if self.current_player_id == 2 else 2
    
    def game_end(self):
        """一共三种状态: 红赢, 黑赢, 和棋"""
        if self.winner is not None:
            return True, self.winner
        # 先手判负: 若2次吃子间隔的回合数大于等于阈值, 则后手判胜
        elif self.kill_action >= CONFIG["kill_action"]:
            return True, self.backhand_player
        return False, -1
    
    def get_current_player_color(self):
        return self.current_player_color
    
    def get_current_player_id(self):
        return self.current_player_id

class Game(object):
    def __init__(self, board: Board):
        self.board = board
    
    def graphic(self, board: Board, player1_color, player2_color):
        print("player1 take:", player1_color)
        print("player2 take:", player2_color)
        print_board(state_list2state_array(board.state_deque[-1]))

    # 等实现 MCTSPlayer 类后再看
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        """
        :param start_player: 先手玩家 ID
        """
        if start_player not in (1, 2):
            raise Exception("start_player must be either 1 or 2")
        self.board.init_board(start_player)
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        
        while True:
            current_player = self.board.get_current_player_id()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                print("Game end. Winner is ", players[winner])
                return winner

    def start_self_play(self, player, is_shown=False, temp=1e-3):
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


if __name__ == '__main__':
    # _array = np.array([1, 0, 0, 0, 0, 0, 0])
    # print(array2string(_array))

    # new_state = change_state(state_list_init, move='0010')
    # for row in range(10):
    #     print(new_state[row])

    # _state_list = copy.deepcopy(state_list_init)
    # print_board(state_list2state_array(_state_list))

    move_id2move_action, move_action2move_id = get_all_legal_moves()
    print(len(move_id2move_action), len(move_action2move_id))

    # print(flip_map('0122'))

    # moves = get_legal_moves(state_deque_init, '黑')
    # print(len(moves))
    # print([move_id2move_action[id] for id in moves])

    moves = get_legal_moves(state_deque_init, '黑')
    print(len(moves))
    print([move_id2move_action[id] for id in moves])

    # moves = get_legal_moves(state_deque_init, '红')
    # print(len(moves))

    # 红
    print([move_id2move_action[id] for id in moves] == ['0010', '0020', '0120', '0122', '0224', '0220', '0314', '0414', '0514', '0628', '0624', '0726', '0728', '0818', '0828', '2120', '2122', '2123', '2124', '2125', '2126', '2111', '2131', '2141', '2151', '2161', '2191', '2726', '2725', '2724', '2723', '2722', '2728', '2717', '2737', '2747', '2757', '2767', '2797', '3040', '3242', '3444', '3646', '3848'])
    # 黑
    print([move_id2move_action[id] for id in moves] == ['6050', '6252', '6454', '6656', '6858', '7170', '7172', '7173', '7174', '7175', '7176', '7161', '7151', '7141', '7131', '7101', '7181', '7776', '7775', '7774', '7773', '7772', '7778', '7767', '7757', '7747', '7737', '7707', '7787', '9080', '9070', '9170', '9172', '9270', '9274', '9384', '9484', '9584', '9674', '9678', '9776', '9778', '9888', '9878'])
