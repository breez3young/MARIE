import ray
import wandb
from copy import deepcopy

from agent.workers.DreamerWorker import DreamerWorker
import ipdb

import numpy as np
import pickle
from pathlib import Path
from environments import Env

class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        ray.init()

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]
        self.env_type = controller_config.ENV_TYPE
        
        eval_controller_config = deepcopy(controller_config)
        eval_controller_config.temperature = 1.0  # 1.0
        if hasattr(eval_controller_config, 'determinisitc'):
            eval_controller_config.determinisitc = True

        self.eval_episodes_num = 10
        self.eval_workers = [DreamerWorker.remote(i, env_config, eval_controller_config) for i in range(self.eval_episodes_num)]
        self.eval_tasks = []

    def append(self, idx, update):
        self.tasks.append(self.workers[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs
    
    ## eval
    def eval_append(self, idx, update):
        self.eval_tasks.append(self.eval_workers[idx].run.remote(update))
        
    def evaluate(self, model_params):
        eval_win_rate = 0.
        eval_returns = 0.
        eval_steps = 0.
        
        for i in range(self.eval_episodes_num):
            self.eval_append(i, model_params)

        for i in range(self.eval_episodes_num):
            # self.eval_append(i, model_params)
            done_id, eval_tasks = ray.wait(self.eval_tasks)
            
            self.eval_tasks = eval_tasks
            eval_rollout, eval_info = ray.get(done_id)[0]
            
            eval_win_rate += eval_info["reward"] if eval_info["reward"] is not None else 0.
            eval_returns += eval_rollout["reward"].sum(0).mean()
            eval_steps += eval_info["steps_done"]
        
        return eval_win_rate / self.eval_episodes_num, eval_returns / self.eval_episodes_num, eval_steps / self.eval_episodes_num


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

        self.save_path = Path(learner_config.RUN_DIR).parent / f"marie_{learner_config.map_name}_seed{learner_config.seed}.pkl"
        self.env_type = controller_config.ENV_TYPE
        
    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10, save_interval= 10000, save_mode="interval"):
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0
        last_eval_steps = 0
        
        eval_win_rates = []
        eval_ret_list  = []
        steps = []

        wandb.define_metric("steps")
        wandb.define_metric("reward", step_metric="steps")
        wandb.define_metric("eval_win_rate", step_metric="steps")
        wandb.define_metric("eval_returns", step_metric="steps")

        while True:
            # NOTE: array manager backend... mp
            rollout, info = self.server.run()
            ent = rollout['entropy'].mean(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            if self.env_type == Env.STARCRAFT:
                wandb.log({'win': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Win: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')
            elif self.env_type == Env.SMAX:
                wandb.log({'win': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Win: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')
            elif self.env_type == Env.MAMUJOCO or self.env_type == Env.PETTINGZOO:
                wandb.log({'rew_per_step': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Rew per step: {info["reward"]}', 'Returns: %.4f' % returns, f"Average std: {ent_str}", sep=' | ')
            else:
                wandb.log({'scores': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Scores: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')


            wandb.log({'returns': returns, "episodes": cur_episode})

            self.learner.step(rollout)

            ## save model
            if (save_interval_steps - last_save_steps) > save_interval and save_mode == "interval":
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 1000}Ksteps.pth")
                last_save_steps = save_interval_steps // save_interval * save_interval

            ## evaluation
            if (save_interval_steps - last_eval_steps) > 500:
                eval_win_rate, eval_returns, aver_eval_steps = self.server.evaluate(self.learner.params())
                last_eval_steps = save_interval_steps // 500 * 500
                
                wandb.log({'eval_win_rate': eval_win_rate, "steps": save_interval_steps})
                wandb.log({'eval_returns': eval_returns, "steps": save_interval_steps})

                steps.append(save_interval_steps)
                eval_win_rates.append(eval_win_rate)
                eval_ret_list.append(eval_returns)

                # Save PKL data immediately after each evaluation
                stored_dict = {
                    'steps': np.array(steps),
                    'eval_win_rates': np.array(eval_win_rates),
                    'eval_returns': np.array(eval_ret_list),
                }
                with open(self.save_path, 'wb') as f:
                    pickle.dump(stored_dict, f)

                if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
                    print(f"Steps: {save_interval_steps}, Eval_win_rate: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                elif self.env_type == Env.MAMUJOCO or self.env_type == Env.PETTINGZOO:
                    print(f"Steps: {save_interval_steps}, Eval rew per step: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                else:
                    print(f"Steps: {save_interval_steps}, Eval average scores: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_final.pth")
                # self.learner.visualize_attention_map(-1, save_mode='final')
                break
            
            self.server.append(info['idx'], self.learner.params())
    
    # only train the actor and critic
    def train_actor(self, world_model_path, max_steps=10 ** 10, max_episodes=10 ** 10):
        ## preload world model
        self.learner.load_pretrained_wm(world_model_path)
        
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0

        wandb.define_metric("steps")
        wandb.define_metric("win rate", step_metric="steps")
        
        while True:
            rollout, info = self.server.run()
            ent = rollout['entropy'].sum(0) / (rollout['entropy'] > 1e-6).sum(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            wandb.log({'win rate': info["reward"], 'steps': cur_steps})
            wandb.log({'returns': returns, "episodes": cur_episode})

            print("%4s" % cur_episode, "%5s" % (cur_steps), info["reward"], 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')

            # train actor only
            self.learner.train_actor_only(rollout)

            if (save_interval_steps - last_save_steps) > 10000:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 10000}Ksteps.pth")
                last_save_steps = save_interval_steps // 10000 * 10000

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            
            self.server.append(info['idx'], self.learner.params())
        
        