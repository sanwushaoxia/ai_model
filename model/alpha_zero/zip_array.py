import numpy as np

def zip_array(array: np.array, data=0.):
    """
    压缩二维数组
    array: 被压缩的二维数组
    """
    zip_res = []
    zip_res.append([len(array), len(array[0]), 0])
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] != data:
                zip_res.append([i, j, array[i][j]])
    return np.array(zip_res)

def recovery_array(array, data=0.):
    """
    解压二维数组
    array: 被解压的二维数组
    """
    recovery_res = []
    # array[0][0] 为二维数组的行数
    for i in range(int(array[0][0])):
        # array[0][1] 为二维数组的列数
        recovery_res.append([data for _ in range(int(array[0][1]))])
    for i in range(1, len(array)):
        # array[i][0] 为行指标, array[i][1] 为列指标, array[i][2] 为值
        recovery_res[int(array[i][0])][int(array[i][1])] = array[i][2]
    return np.array(recovery_res)

def zip_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = state.reshape((9, -1))
    mcts_prob = mcts_prob.reshape((2, -1))
    state = zip_array(state)
    mcts_prob = zip_array(mcts_prob)
    return state, mcts_prob, winner

def recovery_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = recovery_array(state)
    mcts_prob = recovery_array(mcts_prob)
    state = state.reshape((9, 10, 9))
    mcts_prob = mcts_prob.reshape(2086)
    return state, mcts_prob, winner

if __name__ == "__main__":
    I = np.eye(3)
    print(zip_array(I))
    print(recovery_array(zip_array(I)))
