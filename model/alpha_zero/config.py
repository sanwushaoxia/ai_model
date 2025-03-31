CONFIG = {
    "kill_action": 30,     # 和棋回合数, 当前实现判和棋为后手获胜
    "dirichlet": 0.2,      # Dirichlet Noise Parameter, 国际象棋 0.3; 日本将棋 0.15; 围棋 0.03
    "play_out": 1200,      # MCTS 每轮探索次数
    "c_puct": 5,           # 权衡当前价值与潜在价值的参数, 其越大代表更看重潜在价值
    "buffer_size": 1000000, # 收集 data_buffer 的数量
    "paddle_model_path": "current_policy.model", # paddle 模型路径
    "pytorch_model_path": "current_policy.pkl",  # pytorch 模型路径
    "train_data_buffer_path": "train_data_buffer.pkl", # 训练数据路径
    "batch_size": 1024,
    "kl_targ": 0.02, # 控制 KL 散度的参数
    "epochs": 5,
    "game_batch_num": 1000, # 训练时, 更新模型的次数
    "use_frame": "pytorch", # 使用的深度学习框架
    "lr": 1e-4, # 学习率
}
