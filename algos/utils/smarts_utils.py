"""
this file contains tuned obs function and reward function
fix ttc calculate
"""
import math

import gym
import numpy as np

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import OGM, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType
from shapely.geometry import LineString

from matplotlib.path import Path

MAX_LANES = 5  # The maximum number of lanes we expect to see in any scenario.
lane_crash_flag = False  # used for training to signal a flipped car
intersection_crash_flag = False  # used for training to signal intersect crash
global_max_len_lane_index = 0
global_max_len_lane = 51
global_lane_ttc = 1.
global_in_genJ = False
global_sample_wp_path = None
global_int_in_gneJ = False
global_last_len_wps_len = 1
threaten_distance = 1.
head_threaten_distance = 1.
teal_threaten_distance = 1.

# ==================================================
# Continous Action Space
# throttle, brake, steering
# ==================================================

ACTION_SPACE = gym.spaces.Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
)

# ==================================================
# Observation Space
# This observation space should match the output of observation(..) below
# ==================================================
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        # To make car follow the waypoints
        # distance from lane center
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # relative heading angle from 10 waypoints in 50 forehead waypoints
        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)),

        "wp_errors": gym.spaces.Box(low=-1e10, high=1e10, shape=(4,)),
        "wp_speed_penalty": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # Car attributes
        # ego speed
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # ego steering
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # To make car learn to slow down, overtake or dodge
        # distance to the closest car in each lane
        "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # time to collide to the closest car in each lane
        "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # ego lane closest social vehicle relative speed
        "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # distance to the closest car in possible intersection direction
        "intersection_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # time to collide to the closest car in possible intersection direction
        "intersection_distance": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # intersection closest social vehicle relative speed
        "closest_its_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # intersection closest social vehicle relative position in vehicle heading coordinate
        "closest_its_nv_rel_pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),

        "min_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),

        "threaten_distance": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "detect_car": gym.spaces.Box(low=-1e10, high=1e10, shape=(40,)),
    }
)


def heading_to_degree(heading):
    # +y = 0 rad. Note the 0 means up direction
    return np.degrees((heading + math.pi) % (2 * math.pi))


def heading_to_vec(heading):
    # axis x: right, y:up
    angle = (heading + math.pi * 0.5) % (2 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)])


def ttc_by_path(ego, wp_paths, neighborhood_vehicle_states, ego_closest_wp):
    global lane_crash_flag
    global intersection_crash_flag

    # init flag, dist, ttc, headings
    lane_crash_flag = False
    intersection_crash_flag = False

    # default 10s
    lane_ttc = np.array([1] * 5, dtype=float)
    # default 100m
    lane_dist = np.array([1] * 5, dtype=float)
    # default 120km/h
    closest_lane_nv_rel_speed = 1

    intersection_ttc = 1
    intersection_distance = 1
    closest_its_nv_rel_speed = 1
    # default 100m
    closest_its_nv_rel_pos = np.array([1, 1])

    # here to set invalid value to 0
    wp_paths_num = len(wp_paths)
    lane_ttc[wp_paths_num:] = 0
    lane_dist[wp_paths_num:] = 0

    # return if no neighbour vehicle or off the routes(no waypoint paths)
    if not neighborhood_vehicle_states or not wp_paths_num:
        return (
            lane_ttc,
            lane_dist,
            closest_lane_nv_rel_speed,
            intersection_ttc,
            intersection_distance,
            closest_its_nv_rel_speed,
            closest_its_nv_rel_pos,
        )

    # merge waypoint paths (consider might not the same length)
    merge_waypoint_paths = []
    for wp_path in wp_paths:
        merge_waypoint_paths += wp_path

    wp_poses = np.array([wp.pos for wp in merge_waypoint_paths])

    # compute neighbour vehicle closest wp
    nv_poses = np.array([nv.position for nv in neighborhood_vehicle_states])
    nv_wp_distance = np.linalg.norm(nv_poses[:, :2][:, np.newaxis] - wp_poses, axis=2)
    nv_closest_wp_index = np.argmin(nv_wp_distance, axis=1)
    nv_closest_distance = np.min(nv_wp_distance, axis=1)

    # get not in same lane id social vehicles(intersect vehicles and behind vehicles)
    wp_lane_ids = np.array([wp.lane_id for wp in merge_waypoint_paths])
    nv_lane_ids = np.array([nv.lane_id for nv in neighborhood_vehicle_states])
    not_in_same_lane_id = nv_lane_ids[:, np.newaxis] != wp_lane_ids
    not_in_same_lane_id = np.all(not_in_same_lane_id, axis=1)

    ego_edge_id = ego.lane_id[1:-2] if ego.lane_id[0] == "-" else ego.lane_id[:-2]
    nv_edge_ids = np.array(
        [
            nv.lane_id[1:-2] if nv.lane_id[0] == "-" else nv.lane_id[:-2]
            for nv in neighborhood_vehicle_states
        ]
    )
    not_in_ego_edge_id = nv_edge_ids[:, np.newaxis] != ego_edge_id
    not_in_ego_edge_id = np.squeeze(not_in_ego_edge_id, axis=1)

    is_not_closed_nv = not_in_same_lane_id & not_in_ego_edge_id
    not_closed_nv_index = np.where(is_not_closed_nv)[0]

    # filter sv not close to the waypoints including behind the ego or ahead past the end of the waypoints
    close_nv_index = np.where(nv_closest_distance < 2)[0]

    if not close_nv_index.size:
        pass
    else:
        close_nv = [neighborhood_vehicle_states[i] for i in close_nv_index]

        # calculate waypoints distance to ego car along the routes
        wps_with_lane_dist_list = []
        for wp_path in wp_paths:
            path_wp_poses = np.array([wp.pos for wp in wp_path])
            wp_poses_shift = np.roll(path_wp_poses, 1, axis=0)
            wps_with_lane_dist = np.linalg.norm(path_wp_poses - wp_poses_shift, axis=1)
            wps_with_lane_dist[0] = 0
            wps_with_lane_dist = np.cumsum(wps_with_lane_dist)
            wps_with_lane_dist_list += wps_with_lane_dist.tolist()
        wps_with_lane_dist_list = np.array(wps_with_lane_dist_list)

        # get neighbour vehicle closest waypoints index
        nv_closest_wp_index = nv_closest_wp_index[close_nv_index]
        # ego car and neighbour car distance, not very accurate since use the closest wp
        ego_nv_distance = wps_with_lane_dist_list[nv_closest_wp_index]

        # get neighbour vehicle lane index
        nv_lane_index = np.array(
            [merge_waypoint_paths[i].lane_index for i in nv_closest_wp_index]
        )

        # get wp path lane index
        lane_index_list = [wp_path[0].lane_index for wp_path in wp_paths]

        for i, lane_index in enumerate(lane_index_list):
            # get same lane vehicle
            same_lane_nv_index = np.where(nv_lane_index == lane_index)[0]
            if not same_lane_nv_index.size:
                continue
            same_lane_nv_distance = ego_nv_distance[same_lane_nv_index]
            closest_nv_index = same_lane_nv_index[np.argmin(same_lane_nv_distance)]
            closest_nv = close_nv[closest_nv_index]
            closest_nv_speed = closest_nv.speed
            closest_nv_heading = closest_nv.heading
            # radius to degree
            closest_nv_heading = heading_to_degree(closest_nv_heading)

            closest_nv_pos = closest_nv.position[:2]
            bounding_box = closest_nv.bounding_box

            # map the heading to make it consistent with the position coordination
            map_heading = (closest_nv_heading + 90) % 360
            map_heading_radius = np.radians(map_heading)
            nv_heading_vec = np.array(
                [np.cos(map_heading_radius), np.sin(map_heading_radius)]
            )
            nv_heading_vertical_vec = np.array([-nv_heading_vec[1], nv_heading_vec[0]])

            # get four edge center position (consider one vehicle take over two lanes when change lane)
            # maybe not necessary
            closest_nv_front = closest_nv_pos + bounding_box.length * nv_heading_vec
            closest_nv_behind = closest_nv_pos - bounding_box.length * nv_heading_vec
            closest_nv_left = (
                closest_nv_pos + bounding_box.width * nv_heading_vertical_vec
            )
            closest_nv_right = (
                closest_nv_pos - bounding_box.width * nv_heading_vertical_vec
            )
            edge_points = np.array(
                [closest_nv_front, closest_nv_behind, closest_nv_left, closest_nv_right]
            )

            ep_wp_distance = np.linalg.norm(
                edge_points[:, np.newaxis] - wp_poses, axis=2
            )
            ep_closed_wp_index = np.argmin(ep_wp_distance, axis=1)
            ep_closed_wp_lane_index = set(
                [merge_waypoint_paths[i].lane_index for i in ep_closed_wp_index]
                + [lane_index]
            )

            min_distance = np.min(same_lane_nv_distance)

            if ego_closest_wp.lane_index in ep_closed_wp_lane_index:
                if min_distance < 6:
                    lane_crash_flag = True

                nv_wp_heading = (
                    closest_nv_heading
                    - heading_to_degree(
                        merge_waypoint_paths[
                            nv_closest_wp_index[closest_nv_index]
                        ].heading
                    )
                ) % 360

                # find those car just get from intersection lane into ego lane
                if nv_wp_heading > 30 and nv_wp_heading < 330:
                    relative_close_nv_heading = closest_nv_heading - heading_to_degree(
                        ego.heading
                    )
                    # map nv speed to ego car heading
                    map_close_nv_speed = closest_nv_speed * np.cos(
                        np.radians(relative_close_nv_heading)
                    )
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (map_close_nv_speed - ego.speed) * 3.6 / 120,
                    )
                else:
                    closest_lane_nv_rel_speed = min(
                        closest_lane_nv_rel_speed,
                        (closest_nv_speed - ego.speed) * 3.6 / 120,
                    )

            relative_speed_m_per_s = ego.speed - closest_nv_speed

            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = min_distance / relative_speed_m_per_s
            # normalized into 10s
            ttc /= 10

            for j in ep_closed_wp_lane_index:
                if min_distance / 100 < lane_dist[j]:
                    # normalize into 100m
                    lane_dist[j] = min_distance / 100

                if ttc <= 0:
                    continue

                if j == ego_closest_wp.lane_index:
                    if ttc < 0.1:
                        lane_crash_flag = True

                if ttc < lane_ttc[j]:
                    lane_ttc[j] = ttc

    # get vehicles not in the waypoints lane
    if not not_closed_nv_index.size:
        pass
    else:
        filter_nv = [neighborhood_vehicle_states[i] for i in not_closed_nv_index]

        nv_pos = np.array([nv.position for nv in filter_nv])[:, :2]
        nv_heading = heading_to_degree(np.array([nv.heading for nv in filter_nv]))
        nv_speed = np.array([nv.speed for nv in filter_nv])

        ego_pos = ego.position[:2]
        ego_heading = heading_to_degree(ego.heading)
        ego_speed = ego.speed
        nv_to_ego_vec = nv_pos - ego_pos

        line_heading = (
            (np.arctan2(nv_to_ego_vec[:, 1], nv_to_ego_vec[:, 0]) * 180 / np.pi) - 90
        ) % 360
        nv_to_line_heading = (nv_heading - line_heading) % 360
        ego_to_line_heading = (ego_heading - line_heading) % 360

        # judge two heading whether will intersect
        same_region = (nv_to_line_heading - 180) * (
            ego_to_line_heading - 180
        ) > 0  # both right of line or left of line
        ego_to_nv_heading = ego_to_line_heading - nv_to_line_heading
        valid_relative_angle = (
            (nv_to_line_heading - 180 > 0) & (ego_to_nv_heading > 0)
        ) | ((nv_to_line_heading - 180 < 0) & (ego_to_nv_heading < 0))

        # emit behind vehicles
        valid_intersect_angle = np.abs(line_heading - ego_heading) < 90

        # emit patient vehicles which stay in the intersection
        not_patient_nv = nv_speed > 0.01

        # get valid intersection sv
        intersect_sv_index = np.where(
            same_region & valid_relative_angle & valid_intersect_angle & not_patient_nv
        )[0]

        if not intersect_sv_index.size:
            pass
        else:
            its_nv_pos = nv_pos[intersect_sv_index][:, :2]
            its_nv_speed = nv_speed[intersect_sv_index]
            its_nv_to_line_heading = nv_to_line_heading[intersect_sv_index]
            line_heading = line_heading[intersect_sv_index]
            # ego_to_line_heading = ego_to_line_heading[intersect_sv_index]

            # get intersection closest vehicle
            ego_nv_distance = np.linalg.norm(its_nv_pos - ego_pos, axis=1)
            ego_closest_its_nv_index = np.argmin(ego_nv_distance)
            ego_closest_its_nv_distance = ego_nv_distance[ego_closest_its_nv_index]

            line_heading = line_heading[ego_closest_its_nv_index]
            ego_to_line_heading = (
                heading_to_degree(ego_closest_wp.heading) - line_heading
            ) % 360

            ego_closest_its_nv_speed = its_nv_speed[ego_closest_its_nv_index]
            its_closest_nv_to_line_heading = its_nv_to_line_heading[
                ego_closest_its_nv_index
            ]
            # rel speed along ego-nv line
            closest_nv_rel_speed = ego_speed * np.cos(
                np.radians(ego_to_line_heading)
            ) - ego_closest_its_nv_speed * np.cos(
                np.radians(its_closest_nv_to_line_heading)
            )
            closest_nv_rel_speed_m_s = closest_nv_rel_speed
            if abs(closest_nv_rel_speed_m_s) < 1e-5:
                closest_nv_rel_speed_m_s = 1e-5
            ttc = ego_closest_its_nv_distance / closest_nv_rel_speed_m_s

            intersection_ttc = min(intersection_ttc, ttc / 10)
            intersection_distance = min(
                intersection_distance, ego_closest_its_nv_distance / 100
            )

            # transform relative pos to ego car heading coordinate
            rotate_axis_angle = np.radians(90 - ego_to_line_heading)
            closest_its_nv_rel_pos = (
                np.array(
                    [
                        ego_closest_its_nv_distance * np.cos(rotate_axis_angle),
                        ego_closest_its_nv_distance * np.sin(rotate_axis_angle),
                    ]
                )
                / 100
            )

            closest_its_nv_rel_speed = min(
                closest_its_nv_rel_speed, -closest_nv_rel_speed * 3.6 / 120
            )

            if ttc < 0:
                pass
            else:
                intersection_ttc = min(intersection_ttc, ttc / 10)
                intersection_distance = min(
                    intersection_distance, ego_closest_its_nv_distance / 100
                )

                # if to collide in 3s, make it slow down
                if ttc < 2 or ego_closest_its_nv_distance < 6:
                    intersection_crash_flag = True

    return (
        lane_ttc,
        lane_dist,
        closest_lane_nv_rel_speed,
        intersection_ttc,
        intersection_distance,
        closest_its_nv_rel_speed,
        closest_its_nv_rel_pos,
    )


def ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist):
    # transform lane ttc and dist to make ego lane in the array center

    # index need to be set to zero
    # 4: [0,1], 3:[0], 2:[], 1:[4], 0:[3,4]
    zero_index = [[3, 4], [4], [], [0], [0, 1]]
    zero_index = zero_index[ego_lane_index]

    ttc_by_path[zero_index] = 0
    lane_ttc = np.roll(ttc_by_path, 2 - ego_lane_index)
    lane_dist[zero_index] = 0
    ego_lane_dist = np.roll(lane_dist, 2 - ego_lane_index)

    return lane_ttc, ego_lane_dist


def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center


# ==================================================
# obs function
# ==================================================

"""通过小车的位置，朝向，大小计算出小车四个角的位置
"""
def get_ego_position(ego_position,  
                     ego_heading, 
                     ego_bounding_box):
    # corner
    l, h = ego_bounding_box
    ego_heading_cosine = np.cos(-ego_heading)
    ego_heading_sine = np.sin(-ego_heading)

    ego_left_up_corner = ego_position + np.array([
        -h / 2 * ego_heading_cosine + l / 2 * ego_heading_sine,
        l / 2 * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
    ego_right_up_corner = ego_position + np.array([
        h / 2 * ego_heading_cosine + l / 2 * ego_heading_sine,
        l / 2 * ego_heading_cosine - h / 2 * ego_heading_sine]).reshape(-1)
    ego_left_down_corner = ego_position + np.array([
        -h / 2 * ego_heading_cosine + -l / 2 * ego_heading_sine,
        -l / 2 * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
    ego_right_down_corner = ego_position + np.array([
        h / 2 * ego_heading_cosine + -l / 2 * ego_heading_sine,
        -l / 2 * ego_heading_cosine - +h / 2 * ego_heading_sine]).reshape(-1)

    return np.array([ego_left_up_corner,
                        ego_right_up_corner,
                        ego_right_down_corner,
                        ego_left_down_corner])

"""计算180度扇形范围内离我最近的３辆车的信息 [pos_x, pos_y, speed, heading]
"""
def detect_sector_car(env_obs):
    def _cal_angle(vec):
        if vec[1] < 0:
            base_angle = math.pi
            base_vec = np.array([-1.0, 0.0])
        else:
            base_angle = 0.0
            base_vec = np.array([1.0, 0.0])

        cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
        angle = math.acos(cos)
        return angle + base_angle


    def _get_closest_vehicles(ego, neighbor_vehicles, n):
        ego_pos = ego.position[:2]
        groups = {i: (None, 1e10) for i in range(n)}
        partition_size = math.pi * 2.0 / n
        # get partition
        for v in neighbor_vehicles:
            v_pos = v.position[:2]
            rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
            if np.linalg.norm(rel_pos_vec) < 25.:
                # calculate its partitions
                angle = _cal_angle(rel_pos_vec)
                i = int(angle / partition_size)
                dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
                if dist < groups[i][1]:
                    groups[i] = (v, dist)

        return groups

                   
    ego_state = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states

    surrounding_vehicles = _get_closest_vehicles(ego_state, neighbor_vehicle_states, 8)
    ego_heading_vec = np.array([np.cos(-ego_state.heading), np.sin(-ego_state.heading)])
    
    neareat_vehicle =np.zeros((8, 5), dtype=np.float)
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            continue
        
        v = v[0]
        rel_pos = v.position[:2] - ego_state.position[:2]

        rel_dist = np.sqrt(rel_pos.dot(rel_pos))
        v_heading_vec = np.array([math.cos(-v.heading), math.sin(-v.heading)])

        ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
        rel_pos_norm_2 = rel_pos.dot(rel_pos)
        v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)
        ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
            ego_heading_norm_2 + rel_pos_norm_2
        )

        v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
                v_heading_norm_2 + rel_pos_norm_2
        )

        if ego_cosin <= 0 < v_cosin:
            rel_speed = 0
        else:
            rel_speed = ego_state.speed * ego_cosin - v.speed * v_cosin

        ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)
        if ttc > 10.:
            ttc = 10.
        neareat_vehicle[i, :] = np.array(
            [rel_dist / 60., rel_speed / 120., ttc / 10., rel_pos[0] / 60., rel_pos[1] / 60.]
        )

    return neareat_vehicle.reshape((-1,))

"""过滤保留180度扇形范围内的车的信息
"""
def filter_neighborhood_vehicle_states(env_obs):
    ego_state = env_obs.ego_vehicle_state
    ego_position = ego_state.position[:2]
    ego_heading = ego_state.heading

    vehicle_list = []
    if env_obs.neighborhood_vehicle_states is None:
        return []
    else:
        for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            relative_position = neighborhood_vehicle_position - ego_position
            alpha = np.arctan2(-relative_position[0], relative_position[1])
            if abs(alpha - ego_heading) < np.pi / 2 or abs(alpha - ego_heading) > np.pi * 1.5:
                vehicle_list.append(neighborhood_vehicle_state)

    return vehicle_list

"""检测wp点是否已经与neighborhood小车相交
"""
def point_in_neighbor_vehicle(env_obs, point, neighborhood_vehicle_states):

    if len(neighborhood_vehicle_states) == 0:
        return False
    else:
        for neighborhood_vehicle_state in neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
            neighborhood_vehicle_bounding_box = np.array([neighborhood_vehicle_state.bounding_box.length, neighborhood_vehicle_state.bounding_box.width])

            neighborhood_vehicle_corner_positions = get_ego_position(neighborhood_vehicle_position,
                                                                     neighborhood_vehicle_heading,
                                                                     neighborhood_vehicle_bounding_box,)

            p = Path(neighborhood_vehicle_corner_positions)
            if p.contains_point(point):
                return True
            
    return False


"""检测变道是否安全
"""
def safety_detect(env_obs, target_lane, current_lane):

    def get_backward_wp(pos, heading):
        heading_cosine = np.cos(-heading)
        heading_sine = np.sin(-heading)
        l = [i for i in range(13)]
        h = 1

        wp = []
        for l_ in l:
            wp.append(pos + np.array([
            -h / 2 * heading_cosine + -l_ * heading_sine,
            -l_ * heading_cosine - -h / 2 * heading_sine]).reshape(-1))
            wp.append(pos + np.array([
            h / 2 * heading_cosine + -l_ * heading_sine,
            -l_ * heading_cosine - +h / 2 * heading_sine]).reshape(-1))
        return np.array(wp)
    
    # 如果没有改变赛道，则判定安全
    if target_lane == current_lane:
        return True

    wp_paths = env_obs.waypoint_paths
    if target_lane >= len(wp_paths) or current_lane >= len(wp_paths):
        lane_list = [0]
    else:
        lane_list = [i for i in range(min(target_lane, current_lane), max(target_lane, current_lane)+1)]
        lane_list.remove(current_lane)

    for i in lane_list:
        wp_pos_target_lane = wp_paths[i][0].pos
        wp_heading_target_lane = wp_paths[i][0].heading
        backward_wps = get_backward_wp(wp_pos_target_lane, wp_heading_target_lane)

        vehicle_list = []
        if env_obs.neighborhood_vehicle_states is None:
            return []
        else:
            for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
                neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
                if np.linalg.norm(neighborhood_vehicle_position - wp_pos_target_lane) < 20:
                    vehicle_list.append(neighborhood_vehicle_state)
        
        # 后向检测
        for backward_wp in backward_wps:
            if point_in_neighbor_vehicle(env_obs, backward_wp, vehicle_list):
                return False
        
        # 前向检测
        length = len(wp_paths[i])
        forward_wps = [ wp_paths[i][j].pos for j in range(min(12, length))]
        for forward_wp in forward_wps:
            if point_in_neighbor_vehicle(env_obs, forward_wp, vehicle_list):
                return False

    return True

"""计算离我最近的车的距离，及方向
"""
def get_min_dist(env_obs):

    ego_state = env_obs.ego_vehicle_state
    ego_position = ego_state.position[:2]
    ego_heading = ego_state.heading
    ego_bounding_box = np.array([ego_state.bounding_box.length,
                                 ego_state.bounding_box.width])

    ego_corner = get_ego_position(ego_position, 
                                  ego_heading, 
                                  ego_bounding_box)
    ego_line = LineString([ego_corner[0], 
                           ego_corner[1], 
                           ego_corner[2], 
                           ego_corner[3], 
                           ego_corner[0]])

    min_dist = 5
    min_dist_rleative_heading = 0
    neighborhood_vehicle_states = filter_neighborhood_vehicle_states(env_obs)
    if len(neighborhood_vehicle_states) == 0:
        pass
    else:
        for neighborhood_vehicle_state in neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
            neighborhood_vehicle_bounding_box = np.array([neighborhood_vehicle_state.bounding_box.length, neighborhood_vehicle_state.bounding_box.width])


            if np.linalg.norm(neighborhood_vehicle_position-ego_position) < 10.0:
                neighborhood_vehicle_corner = get_ego_position(neighborhood_vehicle_position, 
                                                       neighborhood_vehicle_heading, 
                                                       neighborhood_vehicle_bounding_box)
        
                neighborhood_vehicle_line = LineString([neighborhood_vehicle_corner[0], 
                                                        neighborhood_vehicle_corner[1], 
                                                        neighborhood_vehicle_corner[2], 
                                                        neighborhood_vehicle_corner[3], 
                                                        neighborhood_vehicle_corner[0]])

                distance = ego_line.distance(neighborhood_vehicle_line)
                if min_dist > distance:
                    relative_position = neighborhood_vehicle_position - ego_position
                    alpha = np.arctan2(-relative_position[0], relative_position[1])
                    if abs(alpha - ego_heading) < np.pi / 3 or abs(alpha - ego_heading) > np.pi * 5 / 3:
                        if (abs(neighborhood_vehicle_heading - ego_heading) > 0.2 and abs(neighborhood_vehicle_heading - ego_heading) < (np.pi*2 - 0.2)) or \
                            (ego_state.lane_id == neighborhood_vehicle_state.lane_id):
                            min_dist = distance
                            min_dist_rleative_heading = alpha

    return min_dist, min_dist_rleative_heading


"""计算每辆车的意图
"""
def threaten_via_intent(env_obs, max_len_index):
    def get_neighborhood_wp(pos, heading, speed):
        heading_cosine = np.cos(-heading)
        heading_sine = np.sin(-heading)

        # 前向计算
        l = [i for i in range(15+int(speed*2))]
        h = 0.8

        wp = []
        for l_ in l:
            wp.append((pos + np.array([
                -h / 2 * heading_cosine + l_ / 2 * heading_sine,
                l_ / 2 * heading_cosine - -h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))
            wp.append((pos + np.array([
                h / 2 * heading_cosine + l_ / 2 * heading_sine,
                l_ / 2 * heading_cosine - h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))
        
        # 后向计算(为了报这个车的size包括进来)
        l = [i for i in range(1, 4)]
        for l_ in l:
            wp.append((pos + np.array([
                -h / 2 * heading_cosine + -l_ / 2 * heading_sine,
                -l_ / 2 * heading_cosine - -h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))
            wp.append((pos + np.array([
                h / 2 * heading_cosine + -l_ / 2 * heading_sine,
                -l_ / 2 * heading_cosine - h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))
            
        return np.concatenate(wp, axis=0)
    
    def get_ego_forward_wp(wps, max_i=20):
        l = 0.05
        h = 0.8

        forward_wps = []
        i = 1
        for wp in wps:
            heading_cosine = np.cos(-wp.heading)
            heading_sine = np.sin(-wp.heading)
            pos = wp.pos

            forward_wps.append((pos + np.array([
                -h / 2 * heading_cosine + l / 2 * heading_sine,
                l / 2 * heading_cosine - -h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))
            forward_wps.append((pos + np.array([
                h / 2 * heading_cosine + l / 2 * heading_sine,
                l / 2 * heading_cosine - h / 2 * heading_sine]).reshape(-1)).reshape(1, -1))

            i += 1
            if i > max_i:
                break
        
        return np.concatenate(forward_wps, axis=0)

    def get_min_value_and_index(matrix, limit=2.0):
        shape = matrix.shape[0]
        for i in range(shape):
            if matrix[i] < limit:
                return matrix[i], i // 2
            
            if i == shape-1:
                return 10, 20


    ego_state = env_obs.ego_vehicle_state
    ego_position = ego_state.position[:2]
    ego_heading = ego_state.heading
    if max_len_index >= len(env_obs.waypoint_paths):
        max_len_index = len(env_obs.waypoint_paths) - 1
    ego_path = env_obs.waypoint_paths[max_len_index]
    ego_forward_wps = get_ego_forward_wp(ego_path)

    intend_min_index_min = 20
    intend_min_index_head_min = 20
    intend_min_index_teal_min = 20
    if env_obs.neighborhood_vehicle_states is None:
        return []
    else:
        for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            if np.linalg.norm(neighborhood_vehicle_position - ego_position) < 30:
                """计算同车道的车辆离我的最短距离
                """
                if ego_state.lane_id == neighborhood_vehicle_state.lane_id:
                    relative_position = neighborhood_vehicle_position - ego_position
                    alpha = np.arctan2(-relative_position[0], relative_position[1])

                    # 计算前方最短距离
                    if abs(alpha - ego_heading) < np.pi / 2 or abs(alpha - ego_heading) > np.pi * 1.5:
                        neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
                        neighborhood_vehicle_speed = neighborhood_vehicle_state.speed

                        neighborhood_forward_wps = get_neighborhood_wp(neighborhood_vehicle_position,
                                                                    neighborhood_vehicle_heading,
                                                                    neighborhood_vehicle_speed)
                        
                        intent_distance = np.linalg.norm(ego_forward_wps[:, np.newaxis] - neighborhood_forward_wps, axis=-1)
                        intent_distance = np.min(intent_distance, axis=-1)
                        intend_min_distance, intend_min_index = get_min_value_and_index(intent_distance, limit=2.0)

                        if intend_min_distance < 2.0:
                            if intend_min_index < intend_min_index_head_min:
                                intend_min_index_head_min = intend_min_index 
                    
                    # 计算后方最短距离
                    else:
                        distance = np.linalg.norm(relative_position)
                        if distance < intend_min_index_teal_min:
                            intend_min_index_teal_min = distance

                    continue


                neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
                neighborhood_vehicle_speed = neighborhood_vehicle_state.speed

                neighborhood_forward_wps = get_neighborhood_wp(neighborhood_vehicle_position,
                                                               neighborhood_vehicle_heading,
                                                               neighborhood_vehicle_speed)
                
                intent_distance = np.linalg.norm(ego_forward_wps[:, np.newaxis] - neighborhood_forward_wps, axis=-1)
                intent_distance = np.min(intent_distance, axis=-1)
                intend_min_distance, intend_min_index = get_min_value_and_index(intent_distance, limit=2.0)

                if intend_min_distance < 2.0:
                    # 排除我们后面的车(我们的先验在于如果他在我们后面快撞到我们，那减速也是没有意义的)(但是要除去距离较近的)
                    relative_position = neighborhood_vehicle_position - ego_position
                    alpha = np.arctan2(-relative_position[0], relative_position[1])
                    if not(abs(alpha - ego_heading) < np.pi / 2 or abs(alpha - ego_heading) > np.pi * 1.5):
                        continue

                    # 排除离我跨车道的情况
                    if ego_state.lane_id.split('_')[0] == neighborhood_vehicle_state.lane_id.split('_')[0] \
                    and abs(ego_state.lane_index - neighborhood_vehicle_state.lane_index) > 1:
                        continue
                    
                    # 不动的车排除
                    if neighborhood_vehicle_state.speed < 0.05:
                        continue

                    if intend_min_index < intend_min_index_min:
                        intend_min_index_min = intend_min_index
    
    # 这个数越小越危险
    threaten_level = 1.
    if intend_min_index_min > 20:
        threaten_level = 1.
    else:
        threaten_level = float(intend_min_index_min / 20)

    head_threaten_level = 1.
    if intend_min_index_head_min > 20:
        head_threaten_level = 1.
    else:
        head_threaten_level = float(intend_min_index_head_min / 20)

    teal_threaten_level = 1.
    if intend_min_index_teal_min > 20:
        teal_threaten_level = 1.
    else:
        teal_threaten_level = float(intend_min_index_teal_min / 20)
    
    return threaten_level, head_threaten_level, teal_threaten_level
                
"""计算小车要走的轨迹点
"""
def get_max_index_lane(env_obs):
    global global_max_len_lane_index

    if env_obs.distance_travelled == 0.0:
        global_max_len_lane_index = 0

    wp_paths = env_obs.waypoint_paths
    wp_paths_len = len(wp_paths)

    wps_len = [len(path) for path in wp_paths]
    max_len_lane_index = np.argmax(wps_len)
    max_len = np.max(wps_len)
    max_count = wps_len.count(max_len)

    if max_count == 1:

        if safety_detect(env_obs, max_len_lane_index, env_obs.ego_vehicle_state.lane_index):
            global_max_len_lane_index = max_len_lane_index
            return max_len_lane_index
        else:
            if global_max_len_lane_index >= wp_paths_len:
                global_max_len_lane_index = wp_paths_len - 1
            return global_max_len_lane_index

    # 只算当前车道临近的车道
    max_ids = [i for i, d in enumerate(wps_len) if d == max_len]# if abs(i-env_obs.ego_vehicle_state.lane_index) <= 1 ]
    # 如果最长赛道不止一个，而且都不在我附近，这种情况应该很少出现吧
    if len(max_ids) == 0:
        global_max_len_lane_index = max_len_lane_index
        return max_len_lane_index


    neighborhood_vehicle_states = filter_neighborhood_vehicle_states(env_obs)
    
    for i in range(max_len):
        if len(max_ids) == 1:

            if safety_detect(env_obs, max_ids[0], env_obs.ego_vehicle_state.lane_index):
                global_max_len_lane_index = max_ids[0]
                return max_ids[0]
            else:
                return global_max_len_lane_index

        for max_id in max_ids:
            if point_in_neighbor_vehicle(env_obs, wp_paths[max_id][i].pos, neighborhood_vehicle_states):
                max_ids.remove(max_id)

    if global_max_len_lane_index in max_ids:
        return global_max_len_lane_index
    else:
        i = 1
        while True:
            if min(global_max_len_lane_index+i, wp_paths_len-1) in max_ids:

                if safety_detect(env_obs, global_max_len_lane_index+i, env_obs.ego_vehicle_state.lane_index):
                    global_max_len_lane_index += i 
                    return global_max_len_lane_index
                else:
                    return global_max_len_lane_index

            elif max(global_max_len_lane_index-i, 0) in max_ids:

                if safety_detect(env_obs, global_max_len_lane_index-i, env_obs.ego_vehicle_state.lane_index):
                    global_max_len_lane_index -= i
                    return global_max_len_lane_index
                else:
                    return global_max_len_lane_index
            i += 1

def detect_genJ(sample_wp_path):
    for wp in sample_wp_path:
        if 'gneJ' in wp.lane_id:
            return True
    return False

def detect_int_in_genJ(sample_wp_path, index):
    if index >= len(sample_wp_path):
        index = len(sample_wp_path) - 1


    if 'gneJ' in sample_wp_path[index].lane_id:
        return True
    else:
        return False


from smarts.core.sensors import Observation
from process import *
from map.map import GridMap

cbf_obs = CBFObservation()
linemap = GridMap.load(dir="map", name="mid")
sidemap1 = GridMap.load(dir="map", name="sidemap1")
sidemap2 = GridMap.load(dir="map", name="sidemap2")


def obs_Frenet(obs:Observation):
    cbf_obs(obs)
    return cbf_obs

def observation_adapter(env_obs:Observation):
    """
    Transform the environment's observation into something more suited for your model
    """
    # CBF information
    cbf_obs = obs_Frenet(env_obs)
    position = cbf_obs.ego.p
    cxe, cye, re = linemap.curvature(position[0], position[1], size=[10, 10])
    cbf_obs.r = re
    cbf_obs.cxe = cxe
    cbf_obs.cye = cye
    # linemap.show3circle(
    #     linemap.grid, sidemap1.grid, sidemap2.grid, [cxe, cye], re, cbf_obs
    # )

    min_dist, min_dist_rleative_heading = get_min_dist(env_obs)
    detect_car = detect_sector_car(env_obs)


    global global_last_len_wps_len
    if env_obs.distance_travelled == 0.0:
        global_last_len_wps_len = len(env_obs.waypoint_paths)

    global global_in_genJ
    global_in_genJ = 'gneJ' in env_obs.ego_vehicle_state.lane_id
    # ================================================================
    #   normal info build
    # ================================================================
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth


    ego_lane_index = closest_wp.lane_index

    # ================================================================
    #   lane info build
    # ================================================================
    (
        lane_ttc,
        lane_dist,
        closest_lane_nv_rel_speed,
        intersection_ttc,
        intersection_distance,
        closest_its_nv_rel_speed,
        closest_its_nv_rel_pos,
    ) = ttc_by_path(
        ego_state, wp_paths, env_obs.neighborhood_vehicle_states, closest_wp
    )

    # ================================================================
    #  heading info build
    # ================================================================
    # wp heading errors in current lane in front of vehicle
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])

    # solve case that wps are not enough, then assume the left heading to be same with the last valid.
    wps_len = [len(path) for path in wp_paths]
    max_len_lane_index = get_max_index_lane(env_obs)

    last_wp_index = 0
    for i, wp_index in enumerate(indices):
        if wp_index > wps_len[max_len_lane_index] - 1:
            indices[i:] = last_wp_index
            break
        last_wp_index = wp_index

    global global_sample_wp_path
    global_sample_wp_path = [wp_paths[max_len_lane_index][i] for i in indices]
    heading_errors = [
        math.sin(wp.relative_heading(ego_state.heading)) for wp in global_sample_wp_path
    ]

    # 防止多产生道路实index突变
    global global_max_len_lane_index
    len_wps_len = len(wps_len)
    if len_wps_len > global_last_len_wps_len:
        global_max_len_lane_index = env_obs.ego_vehicle_state.lane_index
    global_last_len_wps_len = len_wps_len

    global global_max_len_lane
    global_max_len_lane = wps_len[max_len_lane_index]
    
    wp_errors = np.array([wp.signed_lateral_error(ego_state.position) for wp in global_sample_wp_path])[:4]
    
    is_genJ = detect_genJ(global_sample_wp_path)

    wp_speed_limit = np.min(np.array([wp.speed_limit for wp in global_sample_wp_path]) / 120.) * 1.065
    
    if is_genJ:
        wp_speed_limit *= 0.9
    if wp_speed_limit > 0.19167:
        wp_speed_limit = 0.19167

    global global_lane_ttc
    lane_ttc, lane_dist = ego_ttc_calc(ego_lane_index, lane_ttc, lane_dist)
    global_lane_ttc = lane_ttc[2]

    global threaten_distance
    global head_threaten_distance
    global teal_threaten_distance
    threaten_distance, head_threaten_distance, teal_threaten_distance = threaten_via_intent(env_obs, env_obs.ego_vehicle_state.lane_index)
    # print(threaten_distance)
    # print(head_threaten_distance)
    # print(teal_threaten_distance)
    # print()
    # print(ego_state.speed)
    global global_int_in_gneJ
    global_int_in_gneJ = detect_int_in_genJ(wp_paths[max_len_lane_index], int(threaten_distance*20)-1)


    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "heading_errors": np.array(heading_errors),
        "wp_errors": wp_errors,
        "wp_speed_penalty": np.array([wp_speed_limit]),
        "speed": np.array([ego_state.speed / 120.]),
        "steering": np.array([ego_state.steering / (0.5 * math.pi)]),
        "lane_ttc": np.array(lane_ttc),
        "lane_dist": np.array(lane_dist),
        "closest_lane_nv_rel_speed": np.array([closest_lane_nv_rel_speed]),
        "intersection_ttc": np.array([intersection_ttc]),
        "intersection_distance": np.array([intersection_distance]),
        "closest_its_nv_rel_speed": np.array([closest_its_nv_rel_speed]),
        "closest_its_nv_rel_pos": np.array(closest_its_nv_rel_pos),
        "min_dist": np.array([min_dist / 5.0, min_dist_rleative_heading/ (np.pi * 2)]),
        "detect_car": detect_car,
        "threaten_distance": np.array([threaten_distance, head_threaten_distance, teal_threaten_distance]),
        "cbf_obs": cbf_obs
    }

# ==================================================
# reward function
# ==================================================
def reward_adapter(env_obs, env_reward):
    """
    Here you can perform your reward shaping.

    The default reward provided by the environment is the increment in
    distance travelled. Your model will likely require a more
    sophisticated reward function
    """
    global lane_crash_flag
    distance_from_center = get_distance_from_center(env_obs)

    center_penalty = -np.abs(distance_from_center)

    # penalise close proximity to lane cars
    if lane_crash_flag:
        crash_penalty = -5
    else:
        crash_penalty = 0

    # penalise close proximity to intersection cars
    if intersection_crash_flag:
        crash_penalty -= 5

    total_reward = np.sum([1.0 * env_reward])
    total_penalty = np.sum([0.1 * center_penalty, 1 * crash_penalty])

    speed = env_obs.ego_vehicle_state.speed

    global global_sample_wp_path
    is_genJ = detect_genJ(global_sample_wp_path)
    wp_speed_limit = np.min(np.array([wp.speed_limit for wp in global_sample_wp_path]) ) * 1.065
    if is_genJ:
        wp_speed_limit *= 0.9
    if wp_speed_limit > 23:
        wp_speed_limit = 23
    speed_penalty = 0.0
    if speed > wp_speed_limit:
         speed_penalty = -1.0 * (speed - wp_speed_limit) * 0.3

    reach_goal_reward = 0.
    if env_obs.events.reached_goal:
        reach_goal_reward = 50.

    # 安全保证
    global threaten_distance
    global head_threaten_distance
    global teal_threaten_distance
    safety_penalty = 0.
    if threaten_distance < 0.56:
        safety_penalty -= (0.56 - threaten_distance) * 1.0
    if head_threaten_distance < 0.5: 
        safety_penalty -= (0.5 - head_threaten_distance) * 1.0
    if teal_threaten_distance < 0.5:
        safety_penalty -= (0.5 - teal_threaten_distance) * 1.0

    # 安全保证2,wps_len小的化，应该减速等车过去然后变道
    global global_max_len_lane

    safety_penalty_2 = 0.
    if global_max_len_lane < 45:
        safety_penalty_2 = speed-4. if speed >= 4. else 0.
    if global_max_len_lane < 30:
        safety_penalty_2 = speed-2. if speed >= 2. else 0.
    if global_max_len_lane < 15:
        safety_penalty_2 = speed
    safety_penalty_2 *= -0.5

    # 安全保证3 lane_ttc要大于一定值，保证安全
    global global_lane_ttc

    safety_penalty_3 = 0.
    if global_lane_ttc < 0.3:
        safety_penalty_3 = (global_lane_ttc - 0.3) * 2.0
  

    # safety_penalty_4 = 0.
    # if threaten_distance < 0.86:
    #     safety_penalty_4 = speed-9. if speed >= 9. else 0.
    # if threaten_distance < 0.66:
    #     safety_penalty_4 = speed-6. if speed >= 6. else 0.
    # if threaten_distance < 0.46:
    #     safety_penalty_4 = speed-3. if speed >= 3. else 0.
    # if threaten_distance < 0.26:
    #     safety_penalty_4 = speed-1. if speed >= 1. else 0.
    # safety_penalty_4 *= -1.0
    # safety_penalty_4 = np.clip(safety_penalty_4, -3, 0)

    return (total_reward + 
            total_penalty + 
            speed_penalty + 
            reach_goal_reward# + 
            #safety_penalty + 
            #safety_penalty_2 + 
            #safety_penalty_3 #+ 
            #safety_penalty_4
            ) / 100.0 # 加一个时间惩罚系数

def action_adapter(model_action):
    assert len(model_action) == 2

    global threaten_distance
    global head_threaten_distance
    throttle = np.clip(model_action[0], 0, 1)
    brake = np.abs(np.clip(model_action[0], -1, 0))

    # global global_in_genJ

    global global_int_in_gneJ
    global global_max_len_lane

    if global_max_len_lane < 20:
        brake = 0.5

    if global_int_in_gneJ:
        if threaten_distance < 0.66:
            throttle = 0.
            brake = 1.
    if threaten_distance < 0.1:
        throttle = 0.
        brake = 1.

    if head_threaten_distance < 0.26:
        throttle = 0.
        brake = 1.

    #print(np.asarray([throttle, brake, model_action[1]]))
    return np.asarray([throttle, brake, model_action[1]])


def info_adapter(observation, reward, info):
    return info


agent_interface = AgentInterface(
    max_episode_steps=None,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    action=ActionSpaceType.Continuous, #LaneWithContinuousSpeed
)

agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)