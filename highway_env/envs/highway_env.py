from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": 0,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        #We apply the action to find different values for each reward function
        rewards = self._rewards(action)
        #then we sum up each reward element to get the final reward value
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        
        #normalize the reward
        #if self.config["normalize_reward"]:
         #   reward = utils.lmap(reward,
          #                      [self.config["collision_reward"], 
           #                      self.rewards["fuel_reward"] + self.config["high_speed_reward"] + self.config["right_lane_reward"]],
            #                    [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    
    def _rewards(self, action: Action) -> Dict[Text, float]:
        m = 1400.04
        ro = 1.206
        s = 2.414
        cx = 0.285
        g = 9.8
        f = 0.02
        i = 5.944
        eta = 0.988
        r = 0.326
        n = 30/3.14*i*self.vehicle.speed/r
        print(self.ac_sahar(action))
        #T = m*r/(i*eta)*(ControlledVehicle.a+1/(2*m)*ro*s*cx*self.vehicle.speed**2+g*f)
        #if T < 0:
         #   F = 0.02975+9.162e-06*n+0.004067*T+ 2.752e-08*n**2+6.902e-06*n*T+0.0004899*T**2
        #elif T >= 0:
         #   F = 1.002-0.0004763*n-0.01355*T+7.58e-08*n**2+8.659e-06*n*T+4.649e-05*T**2  
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
           # "fuel_reward": 1/(F+7.7*self.vehicle.speed/10**5)
        }
      
     
    
    def ac_sahar(self, action) -> None:
        DELTA_SPEED = 5
        target_speed = self.vehicle.speed
        if action == 0 :
            target_speed = self.vehicle.speed     

        if action == 1:
            target_speed = self.vehicle.speed
            
        if action == 2:
            target_speed = self.vehicle.speed
            
        if action == 3:
            target_speed -= DELTA_SPEED
        
        if action == 4:
            target_speed -= DELTA_SPEED

        if action == 5:
            target_speed -= DELTA_SPEED

        if action == 6:
            target_speed += DELTA_SPEED

        if action == 7:
            target_speed += self.DELTA_SPEED     

        if action == 8:
            target_speed += DELTA_SPEED
        acc = ControlledVehicle.speed_control(target_speed)
        return acc
    
    
    
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
