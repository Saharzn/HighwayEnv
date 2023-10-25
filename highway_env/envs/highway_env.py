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
            "reward_speed_range": [15, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "Z": 0
        })
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=35),
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
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"], 
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    
    def fuel(self, action: Action):
        max_fuel_1 = 5
        max_torque = 230
        min_torque = -52
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
        a = self.ac_sahar(action)
        T_unlimited = m*r/(i*eta)*(a+1/(2*m)*ro*s*cx*self.vehicle.speed**2+g*f)
        T = np.clip(T_unlimited, min_torque, max_torque)
        Force = m*(a+1/(2*m)*ro*s*cx*self.vehicle.speed**2+g*f)
        #if T < 0:
         #   F1 = abs(0.02975+9.162e-06*n+0.004067*T+ 2.752e-08*n**2+6.902e-06*n*T+0.0004899*T**2)
        #elif T >= 0:
         #   F1 = 1.002-0.0004763*n-0.01355*T+7.58e-08*n**2+8.659e-06*n*T+4.649e-05*T**2  
        if Force < 0: 
            F1 = abs(1.121-0.2974*self.vehicle.speed-0.00117*Force+0.008702*self.vehicle.speed**2-0.0002958*Force*self.vehicle.speed-3.697e-6*Force**2)
        elif Force >= 0:
            F1 = abs(0.7119-0.147*self.vehicle.speed-0.0002227*Force+0.007622*self.vehicle.speed**2+3.697e-5*Force*self.vehicle.speed+1.704e-8*Force**2)
            
        F2 = -0.0008051*self.vehicle.speed**3+0.05435*self.vehicle.speed**2-1.148*self.vehicle.speed+12.95
        if self.config["Z"] == 0:
            F11 = 0
            F22 = 0
        else:
            F11 = F1
            F22 = F2
        return F11/max_fuel_1
    
    
    
    
    def _rewards(self, action: Action) -> Dict[Text, float]:
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
            #"fuel_reward": self.fuel(action)[0]/20+self.fuel(action)[1]/10
            "fuel_reward": -self.fuel(action)
        }
     
    def ac_sahar(self, action) -> None:
        DELTA_SPEED = 5
        TAU_ACC = 0.6  # [s]
        KP_A = 1 / TAU_ACC 
        MIN_ACCELERATION = -2
        MAX_ACCELERATION = 2
        DEFAULT_TARGET_SPEEDS = np.linspace(10, 30, 2)
        self.target_speeds = DEFAULT_TARGET_SPEEDS
        self.target_speed = self.vehicle.speed
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index) 
        
        if action == 0 :
            self.speed_index = self.speed_to_index(self.vehicle.speed)   

        if action == 1:
            self.speed_index = self.speed_to_index(self.vehicle.speed)
            
        if action == 2:
            self.speed_index = self.speed_to_index(self.vehicle.speed)
            
        if action == 3:
            self.speed_index = self.speed_to_index(self.vehicle.speed) - 1
        
        if action == 4:
            self.speed_index = self.speed_to_index(self.vehicle.speed) - 1

        if action == 5:
            self.speed_index = self.speed_to_index(self.vehicle.speed) - 1

        if action == 6:
            self.speed_index = self.speed_to_index(self.vehicle.speed) + 1

        if action == 7:
            self.speed_index = self.speed_to_index(self.vehicle.speed) + 1   

        if action == 8:
            self.speed_index = self.speed_to_index(self.vehicle.speed) + 1
        
        self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        acc_unlimited = KP_A * (self.target_speed - self.vehicle.speed)
        acc = np.clip(acc_unlimited, MIN_ACCELERATION, MAX_ACCELERATION)
        return acc
       

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (self.target_speeds[-1] - self.target_speeds[0])
        return np.int64(np.clip(np.round(x * (self.target_speeds.size - 1)), 0, self.target_speeds.size - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0])
        return np.int64(np.clip(
            np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)), 0, cls.DEFAULT_TARGET_SPEEDS.size - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))



    
    
    
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
