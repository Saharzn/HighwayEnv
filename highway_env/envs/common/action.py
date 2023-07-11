import functools
import itertools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable, List
from gymnasium import spaces
import numpy as np

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.controller import ControlledVehicle as CV
from highway_env.road.road import Road, LaneIndex, Route

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        The class of a vehicle able to execute the action.

        Must return a subclass of :py:class:`highway_env.vehicle.kinematics.Vehicle`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        Most of the action mechanics are actually implemented in vehicle.act(action), where
        vehicle is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.vehicle_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    def get_available_actions(self):
        """
        For discrete action space, return the list of available actions.
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):
    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    ACTIONS_LAT = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }

    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 speed_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 target_lane_index: LaneIndex = None,
                 **kwargs) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param speed_range: the range of reachable speeds [m/s]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError("Either longitudinal and/or lateral control must be enabled")
        self.dynamical = dynamical
        self.clip = clip

        self.action_lat = self.ACTIONS_LAT if lateral else None
        self.size = 1
        self.action_lat_indexes = {v: k for k, v in self.action_lat.items()}
        self.last_action = [0, 'IDLE']
        self.target_lane_index = target_lane_index 

    def space(self):
        return [spaces.Box(-1., 1., shape=(self.size,), dtype=np.float32), spaces.Discrete(len(self.action_lat))]

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    
    def act(self, action: np.ndarray) -> None:
        if self.clip:
            action[0] = np.clip(action[0], -1, 1)
        if self.speed_range:
            self.controlled_vehicle.MIN_SPEED, self.controlled_vehicle.MAX_SPEED = self.speed_range
           
        if self.longitudinal and self.lateral:
            #if action[1] == "LANE_LEFT":
                #  _from, _to, _id = self.target_lane_index
               #   target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
              #    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
             #        self.target_lane_index = target_lane_index
            #elif action[1] == "LANE_RIGHT":
                  #_from, _to, _id = self.target_lane_index
                  #target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                  #if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                     #self.target_lane_index = target_lane_index
            
            self.controlled_vehicle.act({
                "acceleration": utils.lmap(action['acceleration'], [-1, 1], self.acceleration_range),
                #"steering": steering_control(self,self.target_lane_index),
                "steering":0.1,
            })
        self.last_action = action
    
    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command, -1, 1))
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

def get_available_actions(self) -> List[int]:
        actions = [self.actions_indexes['IDLE']]
        network = self.controlled_vehicle.road.network
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            if l_index[2] < self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_RIGHT'])
        return actions

class DiscreteMetaAction(ActionType):

    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }
    """A mapping of lateral action indexes to labels."""

    def __init__(self,
                 env: 'AbstractEnv',
                 longitudinal: bool = True,
                 lateral: bool = True,
                 target_speeds: Optional[Vector] = None,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else MDPVehicle.DEFAULT_TARGET_SPEEDS
        self.actions = self.ACTIONS_ALL if longitudinal and lateral \
            else self.ACTIONS_LONGI if longitudinal \
            else self.ACTIONS_LAT if lateral \
            else None
        if self.actions is None:
            raise ValueError("At least longitudinal or lateral actions must be included")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: Union[int, np.ndarray]) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        actions = [self.actions_indexes['IDLE']]
        network = self.controlled_vehicle.road.network
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            if l_index[2] < self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_RIGHT'])
        if self.controlled_vehicle.speed_index < self.controlled_vehicle.target_speeds.size - 1 and self.longitudinal:
            actions.append(self.actions_indexes['FASTER'])
        if self.controlled_vehicle.speed_index > 0 and self.longitudinal:
            actions.append(self.actions_indexes['SLOWER'])
        return actions


class MultiAgentAction(ActionType):
    def __init__(self,
                 env: 'AbstractEnv',
                 action_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([action_type.space() for action_type in self.agents_action_types])

    @property
    def vehicle_class(self) -> Callable:
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    def get_available_actions(self):
        return itertools.product(*[action_type.get_available_actions() for action_type in self.agents_action_types])


def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    else:
        raise ValueError("Unknown action type")
