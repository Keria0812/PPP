import pathlib
import pandas as pd
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import DrivingDirectionComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable import EgoIsComfortableStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_acceleration import EgoAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error import EgoExpertL2ErrorStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error_with_yaw import EgoExpertL2ErrorWithYawStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_jerk import EgoLatJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed import EgoMeanSpeedStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import EgoProgressAlongExpertRouteStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import EgoYawRateStatistics
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import PlannerExpertAverageL2ErrorStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress import EgoIsMakingProgressStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_heading_error_within_bound import PlannerExpertAverageHeadingErrorStatistics
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_heading_error_within_bound import PlannerExpertFinalHeadingErrorStatistics
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_l2_error_within_bound import PlannerExpertFinalL2ErrorStatistics
from nuplan.planning.metrics.evaluation_metrics.common.planner_miss_rate_within_bound import PlannerMissRateStatistics
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import SpeedLimitComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import TimeToCollisionStatistics


### Parameters
T = 8 # [s] planning horizon
DT = 0.1# [s] time interval  #plan_changed
LENGTH = get_pacifica_parameters().front_length # [m] vehicle front length
WHEEL_BASE = get_pacifica_parameters().wheel_base # [m] vehicle wheel base
WIDTH = get_pacifica_parameters().width # [m] vehicle width
MAX_LEN = 120 # [m] max length of the path


### Simulation setting
def save_runner_reports(reports, output_dir, report_name):
    """
    Save runner reports to a parquet file in the output directory.
    :param reports: Runner reports returned from each simulation.
    :param output_dir: Output directory to save the report.
    :param report_name: Report name.
    """
    report_dicts = []

    for report in map(lambda x: x.__dict__, reports):  # type: ignore
        if (planner_report := report["planner_report"]) is not None:
            planner_report_statistics = planner_report.compute_summary_statistics()
            del report["planner_report"]
            report.update(planner_report_statistics)
        report_dicts.append(report)

    df = pd.DataFrame(report_dicts)
    df['duration'] = df['end_time'] - df['start_time']

    save_path = pathlib.Path(output_dir) / report_name
    df.to_parquet(save_path)
    print(f'Saved runner reports to {save_path}')


def build_metrics_aggregators(experiment, output_dir, aggregator_metric_dir):
    """
    Build a list of metric aggregators.
    :param cfg: Config
    :return A list of metric aggregators, and the path in which they will  save the results
    """

    aggregator_save_path = f"{output_dir}/{aggregator_metric_dir}"
    aggregator_save_path = pathlib.Path(aggregator_save_path)

    metric_aggregators = []
    metric_aggregator_config = get_aggregator_config(experiment)

    if not aggregator_save_path.exists():
        aggregator_save_path.mkdir(exist_ok=True, parents=True)

    name = metric_aggregator_config[0]
    metric_weights = metric_aggregator_config[1]
    file_name = metric_aggregator_config[2]
    multiple_metrics = metric_aggregator_config[3]
    metric_aggregators.append(WeightedAverageMetricAggregator(name, metric_weights, file_name, aggregator_save_path, multiple_metrics))

    return metric_aggregators


def get_aggregator_config(experiment):
    if experiment == 'open_loop_boxes':
        name = 'open_loop_boxes_weighted_average'
        metric_weights = {'planner_expert_average_l2_error_within_bound': 1, 
                          'planner_expert_average_heading_error_within_bound': 2,
                          'planner_expert_final_l2_error_within_bound': 1, 
                          'planner_expert_final_heading_error_within_bound': 2,
                          'default': 1.0}
        file_name = "open_loop_boxes_weighted_average_metrics"
        multiple_metrics = ['planner_miss_rate_within_bound']
        challenge_name = 'open_loop_boxes'

    elif experiment == 'closed_loop_nonreactive_agents':
        name = 'closed_loop_nonreactive_agents_weighted_average'
        metric_weights = {'ego_progress_along_expert_route': 5.0,
                          'time_to_collision_within_bound': 5.0,
                          'speed_limit_compliance': 4.0,
                          'ego_is_comfortable': 2.0,
                          'default': 1.0}
        file_name = "closed_loop_agents_weighted_average_metrics"
        multiple_metrics = ['no_ego_at_fault_collisions', 'drivable_area_compliance', 
                            'ego_is_making_progress', 'driving_direction_compliance']
        challenge_name = 'closed_loop_nonreactive_agents'
        
    elif experiment == 'closed_loop_reactive_agents':
        name = 'closed_loop_reactive_agents_weighted_average'
        metric_weights = {'ego_progress_along_expert_route': 5.0,
                          'time_to_collision_within_bound': 5.0,
                          'speed_limit_compliance': 4.0,
                          'ego_is_comfortable': 2.0,
                          'default': 1.0}
        file_name = "closed_loop_agents_weighted_average_metrics"
        multiple_metrics = ['no_ego_at_fault_collisions', 'drivable_area_compliance', 
                            'ego_is_making_progress', 'driving_direction_compliance']
        challenge_name = 'closed_loop_reactive_agents'

    else:
        raise TypeError("Experiment type not supported!")

    return name, metric_weights, file_name, multiple_metrics, challenge_name


def get_scenario_map():
    scenario_map = {
        'accelerating_at_crosswalk': [15.0, -3.0],
        'accelerating_at_stop_sign': [15.0, -3.0],
        'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'accelerating_at_traffic_light': [15.0, -3.0],
        'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
        'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
        'behind_bike': [15.0, -3.0],
        'behind_long_vehicle': [15.0, -3.0],
        'behind_pedestrian_on_driveable': [15.0, -3.0],
        'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'changing_lane_to_left': [15.0, -3.0],
        'changing_lane_to_right': [15.0, -3.0],
        'changing_lane_with_lead': [15.0, -3.0],
        'changing_lane_with_trail': [15.0, -3.0],
        'crossed_by_bike': [15.0, -3.0],
        'crossed_by_vehicle': [15.0, -3.0],
        'following_lane_with_lead': [15.0, -3.0],
        'following_lane_with_slow_lead': [15.0, -3.0],
        'following_lane_without_lead': [15.0, -3.0],
        'high_lateral_acceleration': [15.0, -3.0],
        'high_magnitude_jerk': [15.0, -3.0],
        'high_magnitude_speed': [15.0, -3.0],
        'low_magnitude_speed': [15.0, -3.0],
        'medium_magnitude_speed': [15.0, -3.0],
        'near_barrier_on_driveable': [15.0, -3.0],
        'near_construction_zone_sign': [15.0, -3.0],
        'near_high_speed_vehicle': [15.0, -3.0],
        'near_long_vehicle': [15.0, -3.0],
        'near_multiple_bikes': [15.0, -3.0],
        'near_multiple_pedestrians': [15.0, -3.0],
        'near_multiple_vehicles': [15.0, -3.0],
        'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
        'near_pedestrian_on_crosswalk': [15.0, -3.0],
        'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
        'near_trafficcone_on_driveable': [15.0, -3.0],
        'on_all_way_stop_intersection': [15.0, -3.0],
        'on_carpark': [15.0, -3.0],
        'on_intersection': [15.0, -3.0],
        'on_pickup_dropoff': [15.0, -3.0],
        'on_stopline_crosswalk': [15.0, -3.0],
        'on_stopline_stop_sign': [15.0, -3.0],
        'on_stopline_traffic_light': [15.0, -3.0],
        'on_traffic_light_intersection': [15.0, -3.0],
        'starting_high_speed_turn': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_low_speed_turn': [15.0, -3.0],
        'starting_protected_cross_turn': [15.0, -3.0],
        'starting_protected_noncross_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
        'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
        'starting_u_turn': [15.0, -3.0],
        'starting_unprotected_cross_turn': [15.0, -3.0],
        'starting_unprotected_noncross_turn': [15.0, -3.0],
        'stationary': [15.0, -3.0],
        'stationary_at_crosswalk': [15.0, -3.0],
        'stationary_at_traffic_light_with_lead': [15.0, -3.0],
        'stationary_at_traffic_light_without_lead': [15.0, -3.0],
        'stationary_in_traffic': [15.0, -3.0],
        'stopping_at_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_with_lead': [15.0, -3.0],
        'stopping_at_stop_sign_without_lead': [15.0, -3.0],
        'stopping_at_traffic_light_with_lead': [15.0, -3.0],
        'stopping_at_traffic_light_without_lead': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
        'traversing_crosswalk': [15.0, -3.0],
        'traversing_intersection': [15.0, -3.0],
        'traversing_narrow_lane': [15.0, -3.0],
        'traversing_pickup_dropoff': [15.0, -3.0],
        'traversing_traffic_light_intersection': [15.0, -3.0],
        'waiting_for_pedestrian_to_cross': [15.0, -3.0]
    }

    return scenario_map


def get_filter_parameters(num_scenarios_per_type=20, limit_total_scenarios=None, shuffle=True):
    # nuplan challenge
    scenario_types = [
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stopping_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross',
        'behind_long_vehicle',
        'stationary_in_traffic',
        'near_multiple_vehicles',
        'changing_lane',
        'following_lane_with_lead',
    ]
    ''''''
    scenario_tokens = [
        'a36c1b943871552a',
        '88aa2ad613205556',
        '6db7f9f43c655149',
        '45aa9e8713fa5bee',
        '577d4f456cd65460',
        'ac286967ed895963',
        '4836b2dd09895237',
        'f3702a1cc1cb5c64',
        '660d375c109f5eed',
        'fa6b31fc16f251c9',
        '96c975c46cac5a49',
        '8a29aecff22b5657',
        '990551bed2555351',
        '014ad27ed9da5b86',
        'e7b473cea10954cb',
        'beba883bb6285cee',
        'a10ebe68a57c5dda',
        '2754fbd9c2445dce',
        '48d3ee048cff55d6',
        '03d25c49fbc6550e',
        '5d392fa38ff65c3f',
        '59199b4d340558b4',
        'd8a23f0cb78e5938',
        '9ef2cc51be4c51ed',
        'be2efbeede795568',
        '06533b6b947357f7',
        'b87bd5af8ff75bc5',
        'cb0c11da557650db',
        'b6f48c3ca32750d5',
        '96074214b9645952',
        '2e56997063c057a0',
        '217131c79adf588d',
        '7995b9b53ef55e80',
        '4236a1f3c9e35f9d',
        'be8d080bf7cf5835',
        'a61d56930e9b5f36',
        '094db8d50cf95250',
        '442523fcce0950e8',
        'f7700356e6d85410',
        'c6277e74a4f054a7',
        '0d3490827eb45df7',
        '5a79cd2161c754c9',
        'cf87d58064425ee7',
        '1c17b9dad1f65970',
        '93c583b46398560e',
        '4e8af2b28cea5133',
        'ff4d69f4dd0c5474',
        '74914e4a95025d59',
        'e8ec64ceb6d050ff',
        'c0c385cbdd47536b',
        '938b0223319957ef',
        'd13124fa683654b4',
        '32aea40a777c5156',
        '35c3787707ce5deb',
        '5bebf2e252ec5367',
        '6ff0d0bb90d852b9',
        '75025613e3935595',
        '7c34e3a807965fc1',
        '752db4af7e2754e7',
        'a2fc81fb19985d60',
        '6d0875dbcec45b3a',
        '75d56d4c7b6f5013',
        '9a92b5bf8d735034',
        '68e943391ab65773',
        '63352660f02f5e3d',
        '295432d96052578e',
        '8c8e9b7de9bf541a',
        '3e673b48bbf15727',
        'efa032e2bf055c96',
        '78fb87b881535792',
        'd10a8714a31b58b4',
        '45df14109a3a5937',
        'd07309ff567b5953',
        '3f9e713baeec55a9',
        '11f773b0b19055e9',
        '3a0b1987ce79508f',
        'a22b2b1e04595a0e',
        'ae6691cb57175db6',
        '0a2be0a8f9c75775',
        'a9407c1ea38959d5',
        '88ac8fc4c8b25a22',
        '070345fd69165e11',
        '90a02e90433d5137',
        '9c7f7922e3225dec',
        'cb57c23a2cc05786',
        '9c711bed4f175f6d',
        '57441c7185bc5b30',
        '56cdd36746c85788',
        'c5171ce6d7f05d80',
        '8629ab5781b853d3',
        '420653bda4b2575b',
        'edc971e9dc7f5165',
        'd173d3a1c8fb5e8a',
        '2cf6ac2d997a5dff',
        '35265fcdd0be5579',
        'c136ba64909b5ffe',
        '69e91c91e4e65848',
        '840f360aea765a57',
        '88a42c466d2b5ebd',
        'ffbae8b71907545c',
        '15d232cebce05616',
        'be2029b5dc2c5b78',
        'fd1a947d104956f8',
        '63aa10eaed9f53c5',
        'b1cef3eeb5445447',
        'c24af34deb6f54bf',
        '33187fb09d0e52f8',
        '2c5af3c1152c5e69',
        '37947d83063255ea',
        '698cc78af2d154d2',
        'ff02a16ae6c95a42',
        '3454842a96ef511e',
        '61380daa1b275ce0',
        '4f3e807c698a5335',
        '4fde9018255c5f96',
        '6bdb5f343c355f98',
        'c9f1e5d3c8325ac4',
        '2b34e2a1fc6b543c',
        'd63776b80fbd5d4b',
        '56022a77c9fd5a1e',
        '91c272aacfc6511b',
        '904819e474565d84',
        '7f7cd65cacff5f22',
        '1796b59d5f16581f',
        'e0834d617d8d5453',
        '18a8fe4997e053fe',
        '9891d19fd245546a',
        'c88e29a426805b27',
        '71544041acf754bc',
        '72a3637d375155d9',
        '763bb5fb88d2556f',
        '0586b3fb1ffb5fba',
        '0cd6cff135c25fa9',
        '577c248204055c02',
        '1075ba60960c564e',
        '044a3924d9e15bc1',
        '60ad86e2dbf156b5',
        '40e5fc0034035139',
        '067d36d6568657f4',
        '51b40354373a5fac',
        '8e355342a0115145',
        'cbdc3e2a73e15687',
        'eb2ac6cc24f55f6e',
        'd3be3fa904135dd9',
        '338ca19e3fde5381',
        'e10289f39c5f5bea',
        '70ee68fbecd354b0',
        '7c64bfeda7ca5f22',
        '96b881f39d3a55f2',
        '914f752db44d5143',
        'afd9603706985285',
        '88431143abe45625',
        'b703d99845f45666',
        'c921e9258bdf5c0c',
        '66f5844016ec5134',
        'ba267dbeb0c853b5',
        '2b3f110c96995dd1',
        'ae0ace575c775f82',
        'de3cc0dd7ac45a65',
        '230a1a270a105821',
        'c2c649fcd5325b3d',
        '6f2ce5e5530b5e3c',
        '688caa7560c05d03',
        '8de10fd86b825304',
        '61f6e42143cf5eb9',
        '00af7480d144507f',
        'cfa05ee317245c9f',
        '99ef05a2e6a45454',
        'd4edda3ab1d75034',
        '0b7bb38b72ff5034',
        '179a1bf034c650ce',
        '643114dca0825b3f',
        'edbc6fe0dfbf51f1',
        '6fd36832e6925f74',
        '295154d51713573f',
        '40e1f34c92255a0e',
        '1c3d3a5bb86c5d97',
        '6eb38f317f2251f2',
        '0a1adb702c0f5949',
        '003445cf99235331',
        '46236093f497573f',
        'e9faae87fb83540d',
        '2fad34f49b825d6b',
        '521a9f9977d351da',
        '535516116fa35117',
        'd453b49dafae5dd5',
        '89b7bd3592505a26',
        '20f2e6f4bf3f5e80',
        '5fa2710018705849',
        'f9f0880607a15639',
        '9fd5ec2b453d556e',
        '80a05e5b3037536a',
        'ca050f5421925415',
        '4095fd4af9a45b18',
        '33b448fab23a5d03',
        '5cfbb108eb5f584f',
        '2ddc6c9887915bc9',
        '00de3f6da9205a0c',
        'ebf09047e62a5fb8',
        '6639a4f7873a56dc',
        '2703f12098405703',
        '26c1cae14e5b5ce8',
        'a112ea9f3d2b5d5b',
        'a1a3d628e53c5d75',
        'a2ec5056da3c5c67',
        '7289414e82da5b00',
        '1e3f5ab092335059',
        'f15c28ffb0ea5a7a',
        '9b039177b43b5260',
        '7803717ee46c58cc',
        '134e675d2807537f',
        'e569500a38a256bf',
        '6262500c70275443',
        '0e165b03aae35700',
        'b56eda3b0d1c5f67',
        '5bc4f584c6325e50',
        '5cb7891d29545bd4',
        '62ec1599159e5af6',
        'f67f6478ccf65687',
        '0f1513c4f8285ab1',
        '97237e8269415fb0',
        '80ababf8dcfb5914',
        '752323a35a825d22',
        'ba4883a772905c39',
        'b327b4d0432a51a1',
        'e4bc04cbc7eb5940',
        'f8c717165ff15ff7',
        '01010f1fc37b5321',
        '3150573de84e5ea9',
        'c532a0844ee35bde',
        'df68e9d709e65b1e',
        '4570000dd1685c5d',
        'a3b20f60c7835df8',
        'db363d844b3754f1',
        '65d4717562ea5d95',
        'd8834251f07a597f',
        '42ad59cd4b2b5205',
        '698b38a35cc95956',
        'f9ac8947d2c55c1b',
        '0681f7fa37dd5f63',
        '13f3028945475a79',
        'ae6e6bd7567d56bc',
        'd91351e4615859fb',
        '0795bcd734235cf5',
        'd908073d216b5e04',
        'bcf52e9ad12c5ce4',
        '62db9dc16ffd5135',
        'f352e2c4378b5ea1',
        '99e63d494deb5808',
        '3cd758f0d51a55e5',
        '74874f782e725aec',
        '98c66f5373705fd7',
        '0d17d16b86e65700',
        '54da00bc66c7575d',
        'd417ec1ee7295c5f',
        '0488dbdf03b55f00',
        '9e220701888b5ab2',
        '855985a401ab59ea',
        '16a890626fa9570d',
        'daa841dbda985ed0',
        '1e502e7fc8745e0a',
        '9914f87b536b5c23',
        'c2b222d1f0715c00',
        'e94b99e48b355de6',
        '7d365cae2cd45ad8',
        '8181c455582c5623',
        '9e7b8fa4248d55de',
        '159942fd13675580',
        '820bf3685bc955ef',
        '7dde022f98be574c',
        'b8fc7d499e705b68',
        'bc962c4b185859a6',
    ]             # List of scenario tokens to include
    
    #scenario_tokens = None 
    #scenario_tokens =['5ae90b0e5dac5c9d']
    '''
    scenario_tokens = ['136a2f54e24f5895',
                       '262a4b7a1f2c5ab6',
                       '471c1f5d75cb5799',
                       '5da85ef0903e5534',
                       '672d872c54a95c6b',
                       '69e109f6e2a85ab4',
                       '70fb6390fb095016',
                       '8ce5b4c442715fce',
                       '9e30155b8bb55fd9',
                       'aa8237ebd54f5a0b',
                       'c7470ea392845bee',
                       'c94efafdb95f506f',
                       'd0b68e15688c58ad',
                       'f695dbfb3b6f56cb',
                    ]
    '''
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle                             # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


def get_low_level_metrics():
    low_level_metrics = {
        'ego_acceleration': EgoAccelerationStatistics(name='ego_acceleration', category='Dynamics'),
        'ego_expert_L2_error': EgoExpertL2ErrorStatistics(name='ego_expert_L2_error', category='Planning', discount_factor=1),
        'ego_expert_l2_error_with_yaw': EgoExpertL2ErrorWithYawStatistics(name='ego_expert_l2_error_with_yaw', category='Planning', discount_factor=1),
        'ego_jerk': EgoJerkStatistics(name='ego_jerk', category='Dynamics', max_abs_mag_jerk=8.37),
        'ego_lane_change': EgoLaneChangeStatistics(name='ego_lane_change', category='Planning', max_fail_rate=0.3),
        'ego_lat_acceleration': EgoLatAccelerationStatistics(name='ego_lat_acceleration', category='Dynamics', max_abs_lat_accel=4.89),
        'ego_lat_jerk': EgoLatJerkStatistics(name='ego_lat_jerk', category='Dynamics'),
        'ego_lon_acceleration': EgoLonAccelerationStatistics(name='ego_lon_acceleration', category='Dynamics', min_lon_accel=-4.05, max_lon_accel=2.40),
        'ego_lon_jerk': EgoLonJerkStatistics(name='ego_lon_jerk', category='Dynamics', max_abs_lon_jerk=4.13),
        'ego_mean_speed': EgoMeanSpeedStatistics(name='ego_mean_speed', category='Dynamics'),
        'ego_progress_along_expert_route': EgoProgressAlongExpertRouteStatistics(name='ego_progress_along_expert_route', category='Planning', score_progress_threshold=2),
        'ego_yaw_acceleration': EgoYawAccelerationStatistics(name='ego_yaw_acceleration', category='Dynamics', max_abs_yaw_accel=1.93),
        'ego_yaw_rate': EgoYawRateStatistics(name='ego_yaw_rate', category='Dynamics', max_abs_yaw_rate=0.95),
        'planner_expert_average_l2_error_within_bound': PlannerExpertAverageL2ErrorStatistics(name='planner_expert_average_l2_error_within_bound',
                                                                                              category='Planning', metric_score_unit='float',
                                                                                              comparison_horizon=[3, 5, 8], comparison_frequency=1,
                                                                                              max_average_l2_error_threshold=8)
    }

    return low_level_metrics


def get_high_level_metrics(low_level_metrics):
    high_level_metrics = {
        'drivable_area_compliance': DrivableAreaComplianceStatistics(name='drivable_area_compliance',  category='Planning',
                                                                     lane_change_metric=low_level_metrics['ego_lane_change'],
                                                                     max_violation_threshold=0.3, metric_score_unit='bool'),
        'driving_direction_compliance': DrivingDirectionComplianceStatistics(name='driving_direction_compliance', category='Planning',
                                                    lane_change_metric=low_level_metrics['ego_lane_change'], metric_score_unit='bool'),
        'ego_is_comfortable': EgoIsComfortableStatistics(name='ego_is_comfortable', category='Violations', metric_score_unit='bool',
                                                         ego_jerk_metric=low_level_metrics['ego_jerk'],
                                                         ego_lat_acceleration_metric=low_level_metrics['ego_lat_acceleration'],
                                                         ego_lon_acceleration_metric=low_level_metrics['ego_lon_acceleration'],
                                                         ego_lon_jerk_metric=low_level_metrics['ego_lon_jerk'],
                                                         ego_yaw_acceleration_metric=low_level_metrics['ego_yaw_acceleration'],
                                                         ego_yaw_rate_metric=low_level_metrics['ego_yaw_rate']),
        'ego_is_making_progress': EgoIsMakingProgressStatistics(name='ego_is_making_progress', category='Planning', 
                                                                ego_progress_along_expert_route_metric=low_level_metrics['ego_progress_along_expert_route'],
                                                                metric_score_unit='bool', min_progress_threshold=0.2),
        'no_ego_at_fault_collisions': EgoAtFaultCollisionStatistics(name='no_ego_at_fault_collisions', category='Dynamics', metric_score_unit='float',
                                                                    ego_lane_change_metric=low_level_metrics['ego_lane_change']),
        'planner_expert_average_heading_error_within_bound': PlannerExpertAverageHeadingErrorStatistics(name='planner_expert_average_heading_error_within_bound',
                                                        category='Planning', metric_score_unit='float', max_average_heading_error_threshold=0.8,
                                                        planner_expert_average_l2_error_within_bound_metric=low_level_metrics['planner_expert_average_l2_error_within_bound']),
        'planner_expert_final_heading_error_within_bound': PlannerExpertFinalHeadingErrorStatistics(name='planner_expert_final_heading_error_within_bound',
                                                        category='Planning', metric_score_unit='float', max_final_heading_error_threshold=0.8,
                                                        planner_expert_average_l2_error_within_bound_metric=low_level_metrics['planner_expert_average_l2_error_within_bound']),
        'planner_expert_final_l2_error_within_bound': PlannerExpertFinalL2ErrorStatistics(name='planner_expert_final_l2_error_within_bound', category='Planning',
                                                        metric_score_unit='float', max_final_l2_error_threshold=8,
                                                        planner_expert_average_l2_error_within_bound_metric=low_level_metrics['planner_expert_average_l2_error_within_bound']),
        'planner_miss_rate_within_bound': PlannerMissRateStatistics(name='planner_miss_rate_within_bound', category='Planning', metric_score_unit='bool', 
                                                        max_displacement_threshold=[6.0, 8.0, 16.0], max_miss_rate_threshold=0.3,
                                                        planner_expert_average_l2_error_within_bound_metric=low_level_metrics['planner_expert_average_l2_error_within_bound']),
        'speed_limit_compliance': SpeedLimitComplianceStatistics(name='speed_limit_compliance', category='Violations', metric_score_unit='float', 
                                                     max_violation_threshold=1.0, max_overspeed_value_threshold=2.23, lane_change_metric=low_level_metrics['ego_lane_change'])
    }

    high_level_metrics.update({
        'time_to_collision_within_bound': TimeToCollisionStatistics(name='time_to_collision_within_bound', category='Planning', metric_score_unit='bool',
                                                                    time_step_size=0.1, time_horizon=3.0, least_min_ttc=0.95,
                                                                    ego_lane_change_metric=low_level_metrics['ego_lane_change'],
                                                                    no_ego_at_fault_collisions_metric=high_level_metrics['no_ego_at_fault_collisions'])})

    return high_level_metrics


def get_metrics_config(experiment, low_level_metrics, high_level_metrics):
    if experiment == "open_loop_boxes":
        metrics = [low_level_metrics['planner_expert_average_l2_error_within_bound'],
                   high_level_metrics['planner_expert_final_l2_error_within_bound'],
                   high_level_metrics['planner_miss_rate_within_bound'],
                   high_level_metrics['planner_expert_final_heading_error_within_bound'],
                   high_level_metrics['planner_expert_average_heading_error_within_bound']
        ]
    
    elif experiment == 'closed_loop_nonreactive_agents' or experiment == 'closed_loop_reactive_agents':
        metrics = [low_level_metrics['ego_lane_change'], low_level_metrics['ego_jerk'],
                   low_level_metrics['ego_lat_acceleration'], low_level_metrics['ego_lon_acceleration'],
                   low_level_metrics['ego_lon_jerk'], low_level_metrics['ego_yaw_acceleration'],
                   low_level_metrics['ego_yaw_rate'], low_level_metrics['ego_progress_along_expert_route'],
                   high_level_metrics['drivable_area_compliance'], high_level_metrics['no_ego_at_fault_collisions'],
                   high_level_metrics['time_to_collision_within_bound'], high_level_metrics['speed_limit_compliance'],
                   high_level_metrics['ego_is_comfortable'], high_level_metrics['ego_is_making_progress'],
                   high_level_metrics['driving_direction_compliance']
        ]
    
    else:
        raise TypeError("Experiment type not supported!")

    return metrics


def build_metrics_engine(experiment, output_dir, metric_dir):
    main_save_path = pathlib.Path(output_dir) / metric_dir
    low_level_metrics = get_low_level_metrics()
    high_level_metrics = get_high_level_metrics(low_level_metrics)
    selected_metrics = get_metrics_config(experiment, low_level_metrics, high_level_metrics)

    metric_engine = MetricsEngine(main_save_path=main_save_path)
    for metric in selected_metrics:
        metric_engine.add_metric(metric)

    return metric_engine
