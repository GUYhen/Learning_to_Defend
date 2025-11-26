
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv

from xuance import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import IPPO_Agents

from utils.logger import Logger
from environment.xuance_env_wrapper import StackelbergDefenseEnv_XuanCe

def register_stackelberg_env():
    from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
    REGISTRY_MULTI_AGENT_ENV["StackelbergDefense"] = StackelbergDefenseEnv_XuanCe

def create_stackelberg_env_with_config(config):
    return StackelbergDefenseEnv_XuanCe(config=config)

class EpisodeMetricsTracker:

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.metrics_data = []
        self.csv_path = os.path.join(save_dir, 'episode_metrics.csv')

        os.makedirs(save_dir, exist_ok=True)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'defender_reward', 'attacker_reward',
                'network_survivability', 'honeypot_effectiveness',
                'episode_length', 'critical_nodes_compromised',
                'total_compromised', 'attacker_budget_remaining',
                'defender_budget_remaining'
            ])

    def record_episode(self, episode: int, metrics: dict):
        row_data = [
            episode,
            metrics.get('defender_reward', 0.0),
            metrics.get('attacker_reward', 0.0),
            metrics.get('network_survivability', 0.0),
            metrics.get('honeypot_effectiveness', 0.0),
            metrics.get('episode_length', 0),
            metrics.get('critical_nodes_compromised', 0),
            metrics.get('total_compromised', 0),
            metrics.get('attacker_budget_remaining', 0.0),
            metrics.get('defender_budget_remaining', 0.0)
        ]

        self.metrics_data.append(row_data)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def plot_training_curves(self):
        if not self.metrics_data:
            return

        df = pd.DataFrame(self.metrics_data, columns=[
            'episode', 'defender_reward', 'attacker_reward',
            'network_survivability', 'honeypot_effectiveness',
            'episode_length', 'critical_nodes_compromised',
            'total_compromised', 'attacker_budget_remaining',
            'defender_budget_remaining'
        ])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stackelberg Defense Training Progress', fontsize=16)

        axes[0, 0].plot(df['episode'], df['defender_reward'], 'b-', alpha=0.7, label='Defender Reward')
        axes[0, 0].plot(df['episode'], df['defender_reward'].rolling(window=20).mean(), 'r-', linewidth=2, label='20-Episode Moving Average')
        axes[0, 0].set_title('Defender Reward vs Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df['episode'], df['network_survivability'], 'g-', alpha=0.7, label='Network Survivability')
        axes[0, 1].plot(df['episode'], df['network_survivability'].rolling(window=20).mean(), 'r-', linewidth=2, label='20-Episode Moving Average')
        axes[0, 1].set_title('Network Survivability vs Episodes')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Survivability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        axes[1, 0].plot(df['episode'], df['honeypot_effectiveness'], 'orange', alpha=0.7, label='Honeypot Effectiveness')
        axes[1, 0].plot(df['episode'], df['honeypot_effectiveness'].rolling(window=20).mean(), 'r-', linewidth=2, label='20-Episode Moving Average')
        axes[1, 0].set_title('Honeypot Effectiveness vs Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Effectiveness')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].plot(df['episode'], df['defender_reward'], 'b-', alpha=0.7, label='Defender')
        axes[1, 1].plot(df['episode'], df['attacker_reward'], 'r-', alpha=0.7, label='Attacker')
        axes[1, 1].set_title('Rewards Comparison')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

def train_stackelberg_defense_enhanced(critical_nodes_count: int = 10):
    register_stackelberg_env()

    logger = Logger()
    logger.info("="*60)
    logger.info(f"开始XuanCe IPPO Stackelberg防御训练 - {critical_nodes_count}个关键节点")
    logger.info("="*60)

    try:
        configs_dict = get_configs(file_dir="config/ippo_stackelberg_config.yaml")
        configs = argparse.Namespace(**configs_dict)

        configs.critical_nodes_count = critical_nodes_count

        configs.log_dir = f"./logs/ippo_stackelberg_critical_{critical_nodes_count}/"
        configs.model_dir = f"./models/ippo_stackelberg_critical_{critical_nodes_count}/"
        results_dir = f"./results/critical_{critical_nodes_count}/"

        os.makedirs(configs.log_dir, exist_ok=True)
        os.makedirs(configs.model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        metrics_tracker = EpisodeMetricsTracker(results_dir)

        custom_writer = SummaryWriter(os.path.join(configs.log_dir, 'custom_metrics'))

        logger.info(f"创建{configs.parallels}个并行环境")
        logger.info(f"关键节点数量: {critical_nodes_count}")

        configs.critical_nodes_count = critical_nodes_count

        envs = make_envs(configs)

        env_info = envs.env_info
        logger.info(f"智能体数量: {env_info['n_agents']}")
        logger.info(f"观察空间维度: {env_info['obs_shape']}")
        logger.info(f"动作空间维度: {env_info['act_shape']}")

        logger.info(f"深度学习框架: {configs.dl_toolbox}")
        logger.info(f"计算设备: {configs.device}")
        logger.info(f"算法: {configs.agent}")
        logger.info(f"环境: {configs.env_name}")
        logger.info(f"场景: {configs.env_id}")
        logger.info(f"并行环境数: {configs.parallels}")
        logger.info(f"总训练步数: {configs.running_steps}")
        logger.info(f"学习率: {configs.learning_rate}")
        logger.info(f"折扣因子: {configs.gamma}")
        logger.info(f"关键节点数量: {critical_nodes_count}")

        max_episode_steps = configs.horizon_size
        total_episodes = configs.running_steps // (configs.parallels * max_episode_steps)
        logger.info(f"预计训练episodes: {total_episodes}")
        logger.info(f"每个episode最大步数: {max_episode_steps}")

        agent = IPPO_Agents(config=configs, envs=envs)

        logger.info("开始训练...")

        train_steps = configs.running_steps // configs.parallels

        original_step = envs.step
        episode_count = 0
        episode_rewards = {'attacker': 0.0, 'defender': 0.0}
        episode_steps = 0

        def wrapped_step(actions):
            nonlocal episode_count, episode_rewards, episode_steps

            next_obs, rewards, terminateds, truncateds, infos = original_step(actions)

            if isinstance(rewards, list) and len(rewards) > 0:
                reward_dict = rewards[0]
                if isinstance(reward_dict, dict):
                    episode_rewards['attacker'] += reward_dict.get('attacker', 0.0)
                    episode_rewards['defender'] += reward_dict.get('defender', 0.0)
                else:
                    episode_rewards['defender'] += float(reward_dict)
            elif isinstance(rewards, dict):
                episode_rewards['attacker'] += rewards.get('attacker', 0.0)
                episode_rewards['defender'] += rewards.get('defender', 0.0)
            else:
                episode_rewards['defender'] += float(rewards) if rewards is not None else 0.0

            episode_steps += 1

            MAX_EPISODE_LENGTH = 10000
            force_truncate = episode_steps >= MAX_EPISODE_LENGTH

            episode_done = False

            if isinstance(terminateds, list) and len(terminateds) > 0:
                term_val = terminateds[0]
                trunc_val = truncateds[0] if isinstance(truncateds, list) and len(truncateds) > 0 else False

                if isinstance(term_val, dict):
                    episode_done = any(term_val.values()) or (isinstance(trunc_val, dict) and any(trunc_val.values()))
                else:
                    episode_done = bool(term_val) or bool(trunc_val)

            elif isinstance(terminateds, dict):
                episode_done = any(terminateds.values()) or any(truncateds.values())
            else:
                episode_done = bool(terminateds) or bool(truncateds)

            if force_truncate:
                episode_done = True

            if episode_done:
                episode_info = {}
                if isinstance(infos, list) and len(infos) > 0:
                    episode_info = infos[0] if isinstance(infos[0], dict) else {}
                elif isinstance(infos, dict):
                    episode_info = infos

                episode_metrics = {
                    'defender_reward': float(episode_rewards['defender']),
                    'attacker_reward': float(episode_rewards['attacker']),
                    'network_survivability': float(episode_info.get('network_survivability', 0.0)),
                    'honeypot_effectiveness': float(episode_info.get('honeypot_effectiveness', 0.0)),
                    'episode_length': int(episode_steps),
                    'critical_nodes_compromised': int(len(episode_info.get('critical_nodes_status', [])) - sum(episode_info.get('critical_nodes_status', []))),
                    'total_compromised': int(episode_info.get('total_compromised', 0)),
                    'attacker_budget_remaining': float(episode_info.get('attacker_budget_remaining', 0.0)),
                    'defender_budget_remaining': float(episode_info.get('defender_budget_remaining', 0.0))
                }

                metrics_tracker.record_episode(episode_count, episode_metrics)

                custom_writer.add_scalar('Episode/Defender_Reward', episode_metrics['defender_reward'], episode_count)
                custom_writer.add_scalar('Episode/Attacker_Reward', episode_metrics['attacker_reward'], episode_count)
                custom_writer.add_scalar('Episode/Network_Survivability', episode_metrics['network_survivability'], episode_count)
                custom_writer.add_scalar('Episode/Honeypot_Effectiveness', episode_metrics['honeypot_effectiveness'], episode_count)
                custom_writer.add_scalar('Episode/Episode_Length', episode_metrics['episode_length'], episode_count)

                if episode_count % 10 == 0:
                    logger.info(f"Episode {episode_count}: "
                               f"DefReward={episode_metrics['defender_reward']:.3f}, "
                               f"Survivability={episode_metrics['network_survivability']:.3f}, "
                               f"HoneypotEff={episode_metrics['honeypot_effectiveness']:.3f}")

                if episode_count > 0 and episode_count % 100 == 0:
                    logger.info(f"已完成 {episode_count} 个episodes，平均长度: {episode_steps:.1f} 步")

                episode_count += 1
                episode_rewards = {'attacker': 0.0, 'defender': 0.0}
                episode_steps = 0

            return next_obs, rewards, terminateds, truncateds, infos

        envs.step = wrapped_step

        agent.train(train_steps)

        logger.info("训练完成!")

        plot_path = metrics_tracker.plot_training_curves()
        logger.info(f"训练曲线已保存: {plot_path}")

        original_model_dir = f"./models/ippo_stackelberg_critical_{critical_nodes_count}/"
        model_path = os.path.join(original_model_dir, f"final_stackelberg_model_critical_{critical_nodes_count}.pth")

        try:
            agent.save_model(model_path)
            logger.info(f"模型已保存: {model_path}")
        except Exception as e:
            backup_path = f"./final_stackelberg_model_critical_{critical_nodes_count}.pth"
            logger.warning(f"模型保存失败: {e}")
            try:
                agent.save_model(backup_path)
                logger.info(f"模型已保存到备用路径: {backup_path}")
            except Exception as e2:
                logger.error(f"备用路径保存也失败: {e2}")

        custom_writer.close()

        return {
            'config': configs,
            'env_info': env_info,
            'critical_nodes_count': critical_nodes_count,
            'total_episodes': episode_count,
            'csv_path': metrics_tracker.csv_path,
            'plot_path': plot_path
        }

    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

    finally:
        if 'agent' in locals():
            agent.finish()
        if 'envs' in locals():
            envs.close()
        if 'custom_writer' in locals():
            custom_writer.close()
        logger.info("训练资源已释放")

def visualize_enhanced_results(critical_nodes_count: int = 10):
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    logger = Logger()
    logger.info("开始解析并可视化训练结果...")
    
    log_dir = f"logs/ippo_stackelberg_critical_{critical_nodes_count}/custom_metrics"
    
    if not os.path.exists(log_dir):
        logger.error(f"日志目录不存在: {log_dir}")
        return
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Stackelberg Defense Training Results - {critical_nodes_count} Critical Nodes', fontsize=16)
    
    if 'Episode/Defender_Reward' in tags:
        defender_rewards = event_acc.Scalars('Episode/Defender_Reward')
        episodes = [x.step for x in defender_rewards]
        rewards = [x.value for x in defender_rewards]
        
        axes[0].plot(episodes, rewards, 'b-', alpha=0.7)
        axes[0].set_title('Defender Reward')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
    
    if 'Episode/Network_Survivability' in tags:
        survivability = event_acc.Scalars('Episode/Network_Survivability')
        episodes = [x.step for x in survivability]
        values = [x.value for x in survivability]
        
        axes[1].plot(episodes, values, 'g-', alpha=0.7)
        axes[1].set_title('Network Survivability')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Survivability')
        axes[1].grid(True, alpha=0.3)
    
    if 'Episode/Honeypot_Effectiveness' in tags:
        honeypot = event_acc.Scalars('Episode/Honeypot_Effectiveness')
        episodes = [x.step for x in honeypot]
        values = [x.value for x in honeypot]
        
        axes[2].plot(episodes, values, 'r-', alpha=0.7)
        axes[2].set_title('Honeypot Effectiveness')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Effectiveness')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = f"results/critical_{critical_nodes_count}/"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_path = f"{results_dir}/training_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"训练结果图表已保存: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced XuanCe IPPO Stackelberg Defense Training')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'], 
                      default='train', help='运行模式')
    parser.add_argument('--critical_nodes', type=int, choices=[5, 10, 15, 20],
                      default=10, help='关键节点数量')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_stackelberg_defense_enhanced(args.critical_nodes)
    elif args.mode == 'visualize':
        visualize_enhanced_results(args.critical_nodes)
    else:
        print(f"模式 {args.mode} 暂未实现")

if __name__ == "__main__":
    main() 