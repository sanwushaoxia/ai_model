
import copy, time
import numpy as np
from collections import deque
from config import CONFIG

# 初始棋盘
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

state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))

# 字典: 对棋子进行 onehot 编码
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    红马=np.array([0, 1, 0, 0, 0, 0, 0]), 黑马=np.array([0, -1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    红士=np.array([0, 0, 0, 1, 0, 0, 0]), 黑士=np.array([0, 0, 0, -1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    红炮=np.array([0, 0, 0, 0, 0, 1, 0]), 黑炮=np.array([0, 0, 0, 0, 0, -1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]))

# 获取 onehot 编码对应的棋子
def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]

# 返回对当前棋盘 state_list 进行 move 操作后的棋盘
def change_state(state_list, move):
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list

# 对使用 onehot 编码的棋盘进行解码
def print_board(_state_array):
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()

# 对棋盘进行 onehot 编码
def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array

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
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
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

# 检查棋子是否在棋盘上
def check_bounds(toY, toX):
    if toY in range(10) and toX in range(9):
        return True
    return False

# 检查棋子移动是否会遇到阻碍
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

def get_legal_moves(state_deque, current_player_color):
    state_list = state_deque[-1] # 最新棋盘
    old_state_list = state_deque[-4]
    moves = [] # 记录棋盘上所有棋子的合理走法
    face_to_face = False

    k_x = None # 记录黑帅的位置
    k_y = None # 记录黑帅的位置
    K_x = None # 记录红帅的位置
    K_y = None # 记录红帅的位置

    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '一一':
                pass
            else:
                if state_list[y][x] == '黑车' and current_player_color == '黑': # 黑车的合理走法
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    
                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                elif state_list[y][x] == '红车' and current_player_color == '红': # 红车的合理走法
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    
                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                elif state_list[y][x] == '黑马' and current_player_color == '黑': # 黑马的合理走法
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                
                elif state_list[y][x] == '红马' and current_player_color == '红': # 红马的合理走法
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                
                elif state_list[y][x] == '黑象' and current_player_color == '黑': # 黑象的合理走法
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and toY >= 5 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and toY >= 5 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                elif state_list[y][x] == '红象' and current_player_color == '红': # 红象的合理走法
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and toY <= 4 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and toY <= 4 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                elif state_list[y][x] == '黑士' and current_player_color == '黑': # 黑士的合理走法
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                elif state_list[y][x] == '红士' and current_player_color == '红': # 红士的合理走法
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                
                elif state_list[y][x] == '黑帅': # 黑帅的合理走法
                    k_x = x
                    k_y = y
                    if current_player_color == '黑':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                toY = y + i * sign
                                toX = x + (1 - i) * sign

                                if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '黑') and toY >= 7 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)

                elif state_list[y][x] == '红帅': # 红帅的合理走法
                    K_x = x
                    K_y = y
                    if current_player_color == '红':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                toY = y + i * sign
                                toX = x + (1 - i) * sign

                                if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], '红') and toY <= 2 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                
                elif state_list[y][x] == '黑炮' and current_player_color == '黑': # 黑炮的合理走法
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
                                if '红' in state_list[toY][toX]:
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
                                if '红' in state_list[toY][toX]:
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
                                if '红' in state_list[toY][toX]:
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
                                if '红' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                
                elif state_list[y][x] == '红炮' and current_player_color == '红': # 红炮的合理走法
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
                                if '黑' in state_list[toY][toX]:
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
                                if '黑' in state_list[toY][toX]:
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
                                if '黑' in state_list[toY][toX]:
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
                                if '黑' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                elif state_list[y][x] == '黑兵' and current_player_color == '黑': # 黑兵的合理走法
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
                
                elif state_list[y][x] == '红兵' and current_player_color == '红': # 红兵的合理走法
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

    if K_x is not None and k_x is not None and K_x == k_x:
        face_to_face = True
        for i in range(K_y + 1, k_y, 1):
            if state_list[i][K_x] != '一一':
                face_to_face = False
    
    if face_to_face is True:
        if current_player_color == '黑':
            m = str(k_y) + str(k_x) + str(K_y) + str(K_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        else:
            m = str(K_y) + str(K_x) + str(k_y) + str(k_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
    
    moves_id = []
    for move in moves:
        moves_id.append(move_action2move_id[move])
    return moves_id


if __name__ == '__main__':
    # _array = np.array([1, 0, 0, 0, 0, 0, 0])
    # print(array2string(_array))

    # new_state = change_state(state_list_init, move='0010')
    # for row in range(10):
    #     print(new_state[row])

    # _state_list = copy.deepcopy(state_list_init)
    # print_board(state_list2state_array(_state_list))

    # move_id2move_action, move_action2move_id = get_all_legal_moves()
    # print(len(move_id2move_action), len(move_action2move_id))

    # print(flip_map('0122'))

    moves = get_legal_moves(state_deque_init, '黑')
    print(len(moves))
    print(moves)
    print(move_id2move_action[1466])
    print(move_id2move_action[1614])

    moves = get_legal_moves(state_deque_init, '红')
    print(len(moves))
