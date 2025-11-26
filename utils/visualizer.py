import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from environment import StackelbergDefenseEnv
import os

class Visualizer:
    def __init__(self, env: StackelbergDefenseEnv, save_dir: str = "results/plots"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics_history: Dict[str, List], save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if 'defender_reward' in metrics_history:
            self._plot_metric_curve(axes[0], metrics_history['defender_reward'], 
                                  'Defender Reward', 'Episodes', 'Reward', 'blue')
        
        if 'network_survivability' in metrics_history:
            self._plot_metric_curve(axes[1], metrics_history['network_survivability'], 
                                  'Network Survivability', 'Episodes', 'Survivability', 'green')
        
        if 'honeypot_effectiveness' in metrics_history:
            self._plot_metric_curve(axes[2], metrics_history['honeypot_effectiveness'], 
                                  'Honeypot Effectiveness', 'Episodes', 'Effectiveness', 'orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_metric_curve(self, ax, data: List[float], title: str, xlabel: str, ylabel: str, color: str):
        if len(data) == 0:
            return
        
        episodes = list(range(len(data)))
        ax.plot(episodes, data, color=color, alpha=0.3, linewidth=0.5)
        
        window_size = min(100, len(data) // 10)
        if window_size > 1:
            moving_avg = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
            ax.plot(episodes, moving_avg, color=color, linewidth=2, label=f'Moving Average ({window_size})')
            ax.legend()
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    def plot_network_topology_dynamic(self, episode_data: List[Dict], save_path: Optional[str] = None):
        num_snapshots = min(6, len(episode_data))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        pos = nx.spring_layout(self.env.topology.graph, seed=42)
        
        for i in range(num_snapshots):
            idx = i * len(episode_data) // num_snapshots
            data = episode_data[idx]
            
            ax = axes[i]
            self._draw_network_state(ax, pos, data['node_states'], data['attack_intensities'], 
                                   data['defense_intensities'], data.get('honeypot_positions', []))
            ax.set_title(f'Step {idx}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'network_dynamics.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _draw_network_state(self, ax, pos: Dict, node_states: np.ndarray, 
                          attack_intensities: np.ndarray, defense_intensities: np.ndarray,
                          honeypot_positions: List[int]):
        node_colors = []
        node_sizes = []
        
        for i in range(self.env.num_nodes):
            if node_states[i] == 0:
                color = 'red'
                size = 200
            elif i in self.env.critical_nodes:
                color = 'gold'
                size = 300
            elif i in honeypot_positions:
                color = 'purple'
                size = 250
            else:
                color = 'lightblue'
                size = 150
            
            node_colors.append(color)
            node_sizes.append(size)
        
        nx.draw(self.env.topology.graph, pos, ax=ax, node_color=node_colors, 
                node_size=node_sizes, with_labels=True, font_size=8, font_weight='bold')
        
        for i, (x, y) in pos.items():
            if attack_intensities[i] > 0.1:
                attack_circle = patches.Circle((x, y), 0.1 * attack_intensities[i], 
                                             color='red', alpha=0.3, linewidth=2, fill=False)
                ax.add_patch(attack_circle)
            
            if defense_intensities[i] > 0.1:
                defense_circle = patches.Circle((x, y), 0.08 * defense_intensities[i], 
                                              color='blue', alpha=0.3, linewidth=2, fill=False)
                ax.add_patch(defense_circle)
        
        ax.set_title('Network State')
    
    def plot_strategy_heatmap(self, attack_strategy: np.ndarray, defense_strategy: np.ndarray, 
                            save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        attack_matrix = attack_strategy.reshape(-1, 1)
        im1 = ax1.imshow(attack_matrix, cmap='Reds', aspect='auto')
        ax1.set_title('Attack Strategy')
        ax1.set_ylabel('Node Index')
        ax1.set_xlabel('Attack Intensity')
        plt.colorbar(im1, ax=ax1)
        
        defense_allocation = defense_strategy[:self.env.num_nodes].reshape(-1, 1)
        honeypot_decisions = defense_strategy[self.env.num_nodes:].reshape(-1, 1)
        
        defense_combined = np.hstack([defense_allocation, honeypot_decisions])
        im2 = ax2.imshow(defense_combined, cmap='Blues', aspect='auto')
        ax2.set_title('Defense Strategy')
        ax2.set_ylabel('Node Index')
        ax2.set_xlabel('Defense | Honeypot')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Defense', 'Honeypot'])
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'strategy_heatmap.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_evaluation_comparison(self, evaluation_results: Dict[str, Dict], 
                                 save_path: Optional[str] = None):
        metrics = ['avg_defender_reward', 'avg_network_survivability']
        methods = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for method in methods:
                if metric in evaluation_results[method]:
                    values.append(evaluation_results[method][metric])
                    labels.append(method.replace('_', ' ').title())
            
            axes[i].bar(labels, values, alpha=0.7)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            
            for j, v in enumerate(values):
                axes[i].text(j, v + max(values) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'evaluation_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_analysis(self, metrics_history: Dict[str, List], 
                                window_size: int = 100, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'defender_reward' in metrics_history:
            self._plot_convergence_metric(axes[0, 0], metrics_history['defender_reward'], 
                                        'Defender Reward Convergence', window_size)
        
        if 'network_survivability' in metrics_history:
            self._plot_convergence_metric(axes[0, 1], metrics_history['network_survivability'], 
                                        'Network Survivability Convergence', window_size)
        
        if 'honeypot_effectiveness' in metrics_history:
            self._plot_convergence_metric(axes[1, 0], metrics_history['honeypot_effectiveness'], 
                                        'Honeypot Effectiveness Convergence', window_size)
        
        if 'defender_reward' in metrics_history and len(metrics_history['defender_reward']) > window_size:
            self._plot_convergence_stability(axes[1, 1], metrics_history['defender_reward'], window_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_convergence_metric(self, ax, data: List[float], title: str, window_size: int):
        if len(data) < window_size:
            return
        
        moving_avg = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
        
        moving_std = pd.Series(data).rolling(window=window_size, min_periods=1).std()
        
        episodes = list(range(len(data)))
        
        ax.plot(episodes, moving_avg, label='Moving Average', linewidth=2)
        ax.fill_between(episodes, moving_avg - moving_std, moving_avg + moving_std, 
                       alpha=0.3, label='±1 STD')
        
        ax.set_title(title)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_stability(self, ax, data: List[float], window_size: int):
        moving_var = pd.Series(data).rolling(window=window_size).var()
        episodes = list(range(len(moving_var)))
        
        ax.plot(episodes, moving_var, color='red', linewidth=2)
        ax.set_title('Training Stability (Moving Variance)')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
        
        if len(moving_var) > 0:
            threshold = np.nanmean(moving_var) * 0.1
            ax.axhline(y=threshold, color='green', linestyle='--', 
                      label=f'Convergence Threshold ({threshold:.3f})')
            ax.legend()
    
    def plot_defense_effectiveness(self, episode_data: Dict, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        degrees = self.env.degrees
        attack_intensities = episode_data.get('final_attack_intensities', np.zeros(self.env.num_nodes))
        
        axes[0, 0].scatter(degrees, attack_intensities, alpha=0.7)
        axes[0, 0].set_xlabel('Node Degree')
        axes[0, 0].set_ylabel('Attack Intensity')
        axes[0, 0].set_title('Attack Intensity vs Node Degree')
        
        critical_degrees = [degrees[i] for i in self.env.critical_nodes]
        critical_attacks = [attack_intensities[i] for i in self.env.critical_nodes]
        axes[0, 0].scatter(critical_degrees, critical_attacks, color='red', s=100, 
                          label='Critical Nodes', alpha=0.8)
        axes[0, 0].legend()
        
        defense_intensities = episode_data.get('final_defense_intensities', np.zeros(self.env.num_nodes))
        
        axes[0, 1].scatter(degrees, defense_intensities, alpha=0.7)
        axes[0, 1].set_xlabel('Node Degree')
        axes[0, 1].set_ylabel('Defense Intensity')
        axes[0, 1].set_title('Defense Allocation vs Node Degree')
        
        axes[1, 0].scatter(attack_intensities, defense_intensities, alpha=0.7)
        axes[1, 0].set_xlabel('Attack Intensity')
        axes[1, 0].set_ylabel('Defense Intensity')
        axes[1, 0].set_title('Attack vs Defense Intensity')
        
        max_intensity = max(max(attack_intensities), max(defense_intensities))
        axes[1, 0].plot([0, max_intensity], [0, max_intensity], 'r--', alpha=0.5, label='Equal Intensity')
        axes[1, 0].legend()
        
        honeypot_positions = episode_data.get('honeypot_positions', [])
        if honeypot_positions:
            honeypot_effectiveness = episode_data.get('honeypot_effectiveness_per_node', np.zeros(self.env.num_nodes))
            
            node_indices = list(range(self.env.num_nodes))
            colors = ['red' if i in honeypot_positions else 'blue' for i in node_indices]
            
            axes[1, 1].scatter(node_indices, honeypot_effectiveness, c=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Node Index')
            axes[1, 1].set_ylabel('Honeypot Effectiveness')
            axes[1, 1].set_title('Honeypot Deployment Effectiveness')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'defense_effectiveness.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_training_report(self, metrics_history: Dict, evaluation_results: Dict, 
                             save_path: Optional[str] = None):
        fig = plt.figure(figsize=(20, 24))
        
        gs1 = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, top=0.95, bottom=0.7)
        
        if 'defender_reward' in metrics_history:
            ax1 = fig.add_subplot(gs1[0, 0])
            self._plot_metric_curve(ax1, metrics_history['defender_reward'], 
                                  'Defender Reward', 'Episodes', 'Reward', 'blue')
        
        if 'network_survivability' in metrics_history:
            ax2 = fig.add_subplot(gs1[0, 1])
            self._plot_metric_curve(ax2, metrics_history['network_survivability'], 
                                  'Network Survivability', 'Episodes', 'Survivability', 'green')
        
        if 'honeypot_effectiveness' in metrics_history:
            ax3 = fig.add_subplot(gs1[1, 0])
            self._plot_metric_curve(ax3, metrics_history['honeypot_effectiveness'], 
                                  'Honeypot Effectiveness', 'Episodes', 'Effectiveness', 'orange')
        
        if 'critical_nodes_compromised' in metrics_history:
            ax4 = fig.add_subplot(gs1[1, 1])
            self._plot_metric_curve(ax4, metrics_history['critical_nodes_compromised'], 
                                  'Critical Nodes Compromised', 'Episodes', 'Count', 'red')
        
        gs2 = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3, top=0.65, bottom=0.4)
        
        if evaluation_results:
            ax5 = fig.add_subplot(gs2[0, 0])
            methods = list(evaluation_results.keys())
            rewards = [evaluation_results[method].get('avg_defender_reward', 0) for method in methods]
            
            bars = ax5.bar(range(len(methods)), rewards, alpha=0.7)
            ax5.set_xticks(range(len(methods)))
            ax5.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
            ax5.set_ylabel('Average Defender Reward')
            ax5.set_title('Method Comparison - Defender Reward')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + max(rewards)*0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            ax6 = fig.add_subplot(gs2[0, 1])
            survivability = [evaluation_results[method].get('avg_network_survivability', 0) for method in methods]
            
            bars = ax6.bar(range(len(methods)), survivability, alpha=0.7, color='green')
            ax6.set_xticks(range(len(methods)))
            ax6.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
            ax6.set_ylabel('Average Network Survivability')
            ax6.set_title('Method Comparison - Network Survivability')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + max(survivability)*0.01,
                        f'{height:.2f}', ha='center', va='bottom')
        
        gs3 = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.3, top=0.35, bottom=0.05)
        ax7 = fig.add_subplot(gs3[0, 0])
        
        pos = nx.spring_layout(self.env.topology.graph, seed=42)
        node_colors = []
        node_sizes = []
        
        for i in range(self.env.num_nodes):
            if i in self.env.critical_nodes:
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append('lightblue')
                node_sizes.append(150)
        
        nx.draw(self.env.topology.graph, pos, ax=ax7, node_color=node_colors, 
                node_size=node_sizes, with_labels=True, font_size=6, font_weight='bold')
        ax7.set_title('Network Topology (Red: Critical Nodes)')
        
        plt.suptitle('Stackelberg Defense Training Report', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_report.png'), dpi=300, bbox_inches='tight')
        
        plt.show() 