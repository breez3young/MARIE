from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.smax.SMAXAgentConfig import SMAXDreamerConfig


class SMAXDreamerLearnerConfig(SMAXDreamerConfig):
    def __init__(self):
        super().__init__()
        # optimal smac config
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4  # 5e-4
        self.VALUE_LR = 5e-4  # 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 1000 # 500
        self.MODEL_EPOCHS = 200 # 60
        self.WM_EPOCHS = 200  # 200
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 30 # 40; 27m bs should be 10, agents_num ~ 10 should be 20
        self.BATCH_SIZE = 30 # 40; 27m bs should be 8, agents_num ~ 10 should be 20
        self.ac_batch_size = 600  # 600
        # self.SEQ_LENGTH = 20
        self.SEQ_LENGTH = self.HORIZON
        
        self.N_SAMPLES = 100  # in 3s_vs_5z, it had better be 200
        self.EPOCHS = 5 # 4; 27m epochs should be 20, agents_num ~ 10 should be 20

        self.TARGET_UPDATE = 20  # 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        self.ENTROPY = 0.001  # with larger 0.01, we can obtain a little bit better performance on 2m_vs_1z
        self.ENTROPY_ANNEALING = 1.0 # 0.99998
        self.GRAD_CLIP_POLICY = 100.0

        # tokenizer
        ## batch size
        self.t_bs = 512
        ## learning rate
        self.t_lr = 1e-4

        # world model
        ## batch size
        self.wm_bs = 64
        ## learning rate
        self.wm_lr = 1e-4 # 5e-4
        self.wm_weight_decay = 0.01

        self.max_grad_norm = 10.0
        
        # debug
        self.is_preload = False
        self.load_path = ""

        self.use_external_rew_model = False

        self.sample_temperature = 'inf'

        ## control whether average the predicted rewards
        self.critic_average_r = False

        ## discrete regression
        self.rewards_prediction_config = {
            'loss_type': 'hlgauss',
            'min_v': -3., 
            'max_v': 3.,
            'bins': 128,
        }

    def create_learner(self):
        return DreamerLearner(self)
