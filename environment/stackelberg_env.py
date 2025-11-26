import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import yaml
from .network_topology import NetworkTopology

class StackelbergDefenseEnv(gym.Env):
    def __init__(self, config_path: str = "config/env_config.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_nodes = self.config['env']['num_nodes']
        self.critical_nodes = self.config['env']['critical_nodes']
        self.max_steps = self.config['env']['max_steps']
        self.epsilon = self.config['network']['epsilon']
        
        self.topology = NetworkTopology(
            num_nodes=self.num_nodes,
            critical_nodes=self.critical_nodes,
            connection_prob=self.config['network']['connection_prob'],
            min_degree=self.config['network']['min_degree'],
            max_degree=self.config['network']['max_degree']
        )
        
        self.adjacency_matrix = self.topology.load_or_generate_topology()
        self.degrees = self.topology.degrees
        self.node_weights = self.topology.node_weights
        
        base_attacker_budget = self.config['attacker']['budget']
        base_defender_budget = self.config['defender']['budget']
        base_critical_nodes = 10

        budget_scale_factor = len(self.critical_nodes) / base_critical_nodes

        self.attacker_budget = base_attacker_budget * budget_scale_factor
        self.defender_budget = base_defender_budget * budget_scale_factor
        self.attacker_unit_costs = np.array(self.config['attacker']['unit_costs'])
        self.defender_unit_costs = np.array(self.config['defender']['unit_costs'])
        self.baseline_success_prob = self.config['attacker']['baseline_success_prob']
        self.cleaning_difficulty = self.config['defender']['cleaning_difficulty']
        self.honeypot_fake_degrees = np.array(self.config['defender']['honeypot_fake_degrees'])
        
        self.reward_weights = self.config['reward_weights']
        
        self.node_states = None
        self.attacker_remaining_budget = None
        self.defender_remaining_budget = None
        self.honeypot_positions = None
        self.current_step = 0
        self.attack_history = []
        self.defense_history = []
        
        self._setup_spaces()
        
        self.metrics = {
            'network_survivability': 0.0,
            'honeypot_effectiveness': 0.0,
            'defender_reward_history': []
        }
    
    def _setup_spaces(self):
        self.attacker_obs_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([float(self.config['network']['max_degree']), 
                          float(self.config['network']['max_degree']), 
                          float(self.num_nodes), 1.0, self.attacker_budget]),
            dtype=np.float32
        )
        
        obs_dim = self.num_nodes * 3 + 2
        self.defender_obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.attacker_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_nodes,), dtype=np.float32
        )
        
        self.defender_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_nodes * 2,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        self.node_states = np.ones(self.num_nodes)
        self.attacker_remaining_budget = self.attacker_budget
        self.defender_remaining_budget = self.defender_budget
        self.honeypot_positions = np.zeros(self.num_nodes, dtype=bool)
        self.current_step = 0
        self.attack_history = []
        self.defense_history = []
        
        self.metrics = {
            'network_survivability': 1.0,
            'honeypot_effectiveness': 0.0,
            'defender_reward_history': []
        }
        
        attacker_obs = self._get_attacker_observation()
        defender_obs = self._get_defender_observation(np.zeros(self.num_nodes))
        
        info = self._get_info()
        
        return {'attacker': attacker_obs, 'defender': defender_obs}, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        attacker_action = actions['attacker']
        defender_action = actions['defender']
        
        attack_intensities = self._process_attacker_action(attacker_action)
        
        defense_intensities, honeypot_deployment = self._process_defender_action(
            defender_action, attack_intensities
        )
        
        attack_results = self._compute_attack_results(attack_intensities, defense_intensities)
        
        self._update_node_states(attack_results)
        
        honeypot_effects = self._compute_honeypot_effects(attack_intensities, honeypot_deployment)
        cleaning_results = self._perform_threat_cleaning(defense_intensities)
        
        attacker_reward = self._compute_attacker_reward(
            attack_intensities, attack_results
        )
        defender_reward = self._compute_defender_reward(
            defense_intensities, honeypot_deployment, honeypot_effects, cleaning_results
        )
        
        self._update_metrics(defender_reward, honeypot_effects)
        
        self.attack_history.append(attack_intensities.copy())
        self.defense_history.append(defense_intensities.copy())
        
        self.current_step += 1
        
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        next_attacker_obs = self._get_attacker_observation()
        next_defender_obs = self._get_defender_observation(attack_intensities)
        
        rewards = {'attacker': attacker_reward, 'defender': defender_reward}
        observations = {'attacker': next_attacker_obs, 'defender': next_defender_obs}
        terminateds = {'attacker': terminated, 'defender': terminated}
        truncateds = {'attacker': truncated, 'defender': truncated}
        info = self._get_info()
        
        return observations, rewards, terminateds, truncateds, info
    
    def _process_attacker_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)
        
        max_intensities = self.attacker_remaining_budget / self.attacker_unit_costs
        attack_intensities = action * np.minimum(max_intensities, 1.0)
        
        cost = np.sum(attack_intensities * self.attacker_unit_costs)
        self.attacker_remaining_budget = max(0, self.attacker_remaining_budget - cost)
        
        return attack_intensities
    
    def _process_defender_action(self, action: np.ndarray, attack_intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        defense_allocation = np.clip(action[:self.num_nodes], 0, 1)
        honeypot_decisions = np.clip(action[self.num_nodes:], 0, 1)
        
        if np.sum(defense_allocation) > 0:
            defense_allocation = defense_allocation / np.sum(defense_allocation)
        
        max_intensities = self.defender_remaining_budget / self.defender_unit_costs
        defense_intensities = defense_allocation * np.minimum(max_intensities, 1.0)
        
        honeypot_deployment = honeypot_decisions > 0.5
        
        defense_cost = np.sum(defense_intensities * self.defender_unit_costs)
        honeypot_cost = np.sum(honeypot_deployment) * 0.5
        total_cost = defense_cost + honeypot_cost
        
        self.defender_remaining_budget = max(0, self.defender_remaining_budget - total_cost)
        self.honeypot_positions = honeypot_deployment
        
        return defense_intensities, honeypot_deployment
    
    def _compute_attack_results(self, attack_intensities: np.ndarray, 
                               defense_intensities: np.ndarray) -> np.ndarray:
        strategic_prob = attack_intensities / (attack_intensities + defense_intensities + self.epsilon)
        
        total_success_prob = (self.baseline_success_prob + 
                            (1 - self.baseline_success_prob) * strategic_prob)
        
        attack_results = np.random.random(self.num_nodes) < total_success_prob
        
        return attack_results
    
    def _update_node_states(self, attack_results: np.ndarray):
        self.node_states[attack_results] = 0
    
    def _compute_honeypot_effects(self, attack_intensities: np.ndarray, 
                                 honeypot_deployment: np.ndarray) -> np.ndarray:
        honeypot_effects = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            if honeypot_deployment[i] and attack_intensities[i] > 0:
                fake_degree = self.honeypot_fake_degrees[i % len(self.honeypot_fake_degrees)]
                deception_prob = fake_degree / (fake_degree + attack_intensities[i] + self.epsilon)
                honeypot_effects[i] = deception_prob
        
        return honeypot_effects
    
    def _perform_threat_cleaning(self, defense_intensities: np.ndarray) -> np.ndarray:
        cleaning_results = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            if self.node_states[i] == 0 and defense_intensities[i] > 0:
                cleaning_prob = min(1.0, defense_intensities[i] / 
                                  (self.cleaning_difficulty * self.attack_history[-1][i] + self.epsilon)
                                  if self.attack_history else defense_intensities[i])
                
                if np.random.random() < cleaning_prob:
                    self.node_states[i] = 1
                    cleaning_results[i] = 1
        
        return cleaning_results
    
    def _compute_attacker_reward(self, attack_intensities: np.ndarray, 
                                attack_results: np.ndarray) -> float:
        damage_reward = np.sum(self.node_weights * attack_results)
        
        attack_cost = np.sum(self.attacker_unit_costs * attack_intensities)
        
        total_reward = (self.reward_weights['attacker']['damage_weight'] * damage_reward - 
                       self.reward_weights['attacker']['cost_weight'] * attack_cost)
        
        return total_reward
    
    def _compute_defender_reward(self, defense_intensities: np.ndarray,
                                honeypot_deployment: np.ndarray, 
                                honeypot_effects: np.ndarray,
                                cleaning_results: np.ndarray) -> float:
        weights = self.reward_weights['defender']
        
        protection_reward = np.sum(self.node_weights * self.node_states)
        
        honeypot_reward = np.sum(honeypot_effects[honeypot_deployment])
        
        cleaning_reward = np.sum(self.node_weights * cleaning_results)
        
        defense_cost = np.sum(self.defender_unit_costs * defense_intensities)
        honeypot_cost = np.sum(honeypot_deployment) * 0.5
        total_cost = defense_cost + honeypot_cost
        
        total_reward = (weights['protection_weight'] * protection_reward + 
                       weights['honeypot_weight'] * honeypot_reward + 
                       weights['cleaning_weight'] * cleaning_reward - 
                       weights['cost_weight'] * total_cost)
        
        return total_reward
    
    def _get_attacker_observation(self) -> np.ndarray:
        avg_degree = np.mean(self.degrees)
        max_degree = np.max(self.degrees)
        compromised_count = np.sum(self.node_states == 0)
        avg_defense = (np.mean(self.defense_history[-1]) 
                      if self.defense_history else 0.0)
        
        return np.array([avg_degree, max_degree, compromised_count, 
                        avg_defense, self.attacker_remaining_budget], dtype=np.float32)
    
    def _get_defender_observation(self, attack_intensities: np.ndarray) -> np.ndarray:
        obs = []
        
        for i in range(self.num_nodes):
            obs.extend([
                attack_intensities[i],
                float(self.node_states[i]),
                self.degrees[i] / np.max(self.degrees)
            ])
        
        obs.append(np.sum(self.honeypot_positions) / self.num_nodes)
        obs.append(self.defender_remaining_budget / self.defender_budget)
        
        return np.array(obs, dtype=np.float32)
    
    def _update_metrics(self, defender_reward: float, honeypot_effects: np.ndarray):
        self.metrics['network_survivability'] = np.sum(self.node_states) / self.num_nodes
        
        self.metrics['honeypot_effectiveness'] = np.mean(honeypot_effects) if np.sum(honeypot_effects) > 0 else 0.0
        
        self.metrics['defender_reward_history'].append(defender_reward)
    
    def _check_termination(self) -> bool:
        critical_compromised = all(self.node_states[i] == 0 for i in self.critical_nodes)
        budget_exhausted = (self.attacker_remaining_budget <= 0 or 
                           self.defender_remaining_budget <= 0)
        
        return critical_compromised or budget_exhausted
    
    def _get_info(self) -> Dict:
        return {
            'step': self.current_step,
            'network_survivability': self.metrics['network_survivability'],
            'honeypot_effectiveness': self.metrics['honeypot_effectiveness'],
            'critical_nodes_status': [self.node_states[i] for i in self.critical_nodes],
            'total_compromised': int(np.sum(self.node_states == 0)),
            'attacker_budget_remaining': self.attacker_remaining_budget,
            'defender_budget_remaining': self.defender_remaining_budget
        }
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Network Survivability: {self.metrics['network_survivability']:.3f}")
            print(f"Honeypot Effectiveness: {self.metrics['honeypot_effectiveness']:.3f}")
            print(f"Critical Nodes Status: {[self.node_states[i] for i in self.critical_nodes]}")
            print(f"Compromised Nodes: {np.sum(self.node_states == 0)}/{self.num_nodes}")
            print("-" * 50)
    
    def get_metrics(self) -> Dict:
        return self.metrics.copy() 