CONFIG = {
    "kill_action": 30,     # 和棋回合数
    "dirichlet": 0.2,      # 国际象棋 0.3; 日本将棋 0.15; 围棋 0.03
    "play_out": 1200,      # 每次移动的模拟次数
    "c_puct": 5,           # u 的权重
    "buffer_size": 100000, # 经验池大小
    "paddle_model_path": "current_policy.model", # paddle 模型路径
    "pytorch_model_path": "current_policy.pkl",  # pytorch 模型路径
    "train_data_buffer_path": "train_data_buffer.pkl", # 训练数据路径
    "batch_size": 512,
    "kl_targ": 0.02, # KL散度
    "epochs": 5,
    "game_batch_num": 3000, # 训练更新的次数
    "use_frame": "pytorch", # 使用的深度学习框架
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
}
