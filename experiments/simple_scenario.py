import numpy as np
import matplotlib.pyplot as plt
from commonocean.scenario.scenario import Scenario, Tag, Location
from commonocean.scenario.waters import Waterway, Shallow, WatersType
from commonocean.scenario.obstacle import StaticObstacle, DynamicObstacle, ObstacleType
from commonocean.scenario.traffic_sign import TrafficSign, TrafficSignElementID, TrafficSignElement
from commonocean.scenario.trajectory import Trajectory
from commonocean.prediction.prediction import TrajectoryPrediction
from commonocean.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonocean.planning.goal import GoalRegion
from commonocean.common.file_writer import CommonOceanFileWriter, OverwriteExistingFile
from commonocean.visualization.draw_dispatch_cr import draw_object

from commonroad.scenario.state import CustomState
from commonroad.geometry.shape import Rectangle, Circle, Polygon
from commonroad.common.util import Interval, AngleInterval

# Design Parameters
dt = 5.0  # 5 second time steps
total_horizon = 15  # 15 time steps (75 seconds total)
benchmark_id = "MAR_Navigation-1_3_T-1"

print("=== Maritime Navigation Scenario Design ===")
print(f"Time step size: {dt} seconds")
print(f"Total horizon: {total_horizon} time steps ({dt * total_horizon} seconds total)")
print(f"Scenario ID: {benchmark_id}")

# Create the scenario
scenario = Scenario(dt, benchmark_id)

# Create waterway network
print("\nCreating waterway network...")
# Main shipping channel (wider waterway for larger vessels)
channel_width = 800  # 800m wide main channel
channel_length = 3000  # 3km long

left_vertices = np.array([[-channel_length / 2, -channel_width / 2],
                          [channel_length / 2, -channel_width / 2]])
right_vertices = np.array([[-channel_length / 2, channel_width / 2],
                           [channel_length / 2, channel_width / 2]])
center_vertices = np.array([[-channel_length / 2, 0.0],
                            [channel_length / 2, 0.0]])

main_waterway_id = scenario.generate_object_id()
main_waterway = Waterway(
    left_vertices,
    center_vertices,
    right_vertices,
    main_waterway_id,
    WatersType.FAIRWAY
)
scenario.add_objects(main_waterway)

# Secondary channel (perpendicular intersection)
secondary_width = 400
secondary_length = 1500

left_vertices_2 = np.array([[0, -secondary_length / 2],
                            [0, secondary_length / 2]])
right_vertices_2 = np.array([[secondary_width, -secondary_length / 2],
                             [secondary_width, secondary_length / 2]])
center_vertices_2 = np.array([[secondary_width / 2, -secondary_length / 2],
                              [secondary_width / 2, secondary_length / 2]])

secondary_waterway = Waterway(left_vertices_2, center_vertices_2, right_vertices_2,
                              scenario.generate_object_id(), WatersType.FAIRWAY)
scenario.add_objects(secondary_waterway)

# Add shallow areas (navigation hazards)
print("Adding shallow water areas...")
# Shallow area 1 - Eastern side
shallow_1_vertices = np.array([[900, -600], [1200, -600], [1200, -300], [900, -300]])
shallow_1 = Shallow(Polygon(shallow_1_vertices), scenario.generate_object_id())
scenario.add_objects(shallow_1)

# Shallow area 2 - Western side  
shallow_2_vertices = np.array([[-1200, 300], [-900, 300], [-900, 600], [-1200, 600]])
shallow_2 = Shallow(Polygon(shallow_2_vertices), scenario.generate_object_id())
scenario.add_objects(shallow_2)

# Add static obstacles (various shapes)
print("Adding static obstacles...")

# Island obstacle (circular)
island_position = np.array([600, 200])
island_obstacle = StaticObstacle(
    scenario.generate_object_id(),
    ObstacleType.LAND,
    Circle(80),  # 80m radius island
    CustomState(position=island_position, orientation=0, time_step=0, velocity=0)
)
scenario.add_objects(island_obstacle)

# Oil platform (rectangular)
platform_position = np.array([-700, -150])
platform_obstacle = StaticObstacle(
    scenario.generate_object_id(),
    ObstacleType.OILRIG,
    Rectangle(length=60, width=40),
    CustomState(position=platform_position, orientation=0.3, time_step=0, velocity=0)
)
scenario.add_objects(platform_obstacle)

# Buoy field (multiple circular buoys)
buoy_positions = [np.array([200, 300]), np.array([250, 320]), np.array([180, 280])]
for i, pos in enumerate(buoy_positions):
    buoy = StaticObstacle(
        scenario.generate_object_id(),
        ObstacleType.BUOY,
        Circle(12),  # 12m radius buoys
        CustomState(position=pos, orientation=0, time_step=0, velocity=0)
    )
    scenario.add_objects(buoy)

# Wreck (irregular polygon shape)
wreck_vertices = np.array([[-300, -400], [-280, -380], [-250, -390], [-260, -420], [-290, -415]])
wreck_obstacle = StaticObstacle(
    scenario.generate_object_id(),
    ObstacleType.ANCHOREDVESSEL,
    Polygon(wreck_vertices),
    CustomState(position=np.array([0, 0]), orientation=0, time_step=0, velocity=0)
)
scenario.add_objects(wreck_obstacle)

# Add navigation aids
print("Adding navigation aids...")
# Port side marker (red)
port_marker = TrafficSign(
    scenario.generate_object_id(),
    [TrafficSignElement(TrafficSignElementID.LATERAL_MARK_RED_A, ["Fl.R 4s"])],
    np.array([-400, -200]),
    virtual=False
)
scenario.add_objects(port_marker, waters_ids=[main_waterway_id])

# Starboard side marker (green)
starboard_marker = TrafficSign(
    scenario.generate_object_id(),
    [TrafficSignElement(TrafficSignElementID.LATERAL_MARK_GREEN_A, ["Fl.G 4s"])],
    np.array([400, 200]),
    virtual=False
)
scenario.add_objects(starboard_marker, waters_ids=[main_waterway_id])

# Create dynamic vessels
print("Creating dynamic vessels...")


def create_vessel_trajectory(start_pos, end_pos, speed_knots, vessel_length):
    """Create a vessel trajectory between two points"""
    # Convert speed from knots to m/s (1 knot = 0.514444 m/s)
    speed_ms = speed_knots * 0.514444

    # Calculate heading
    direction_vector = end_pos - start_pos
    distance = np.linalg.norm(direction_vector)
    heading = np.arctan2(direction_vector[1], direction_vector[0])

    # Initial state
    initial_state = CustomState(
        position=start_pos,
        orientation=heading,
        velocity=speed_ms,
        time_step=0
    )

    # Generate trajectory states
    state_list = []
    total_time = distance / speed_ms
    steps_needed = min(total_horizon - 1, int(total_time / dt))

    for t in range(1, steps_needed + 1):
        # Calculate new position along the path
        progress = (t * dt) / total_time
        if progress >= 1.0:
            new_position = end_pos
        else:
            new_position = start_pos + progress * direction_vector

        new_state = CustomState(
            position=new_position,
            orientation=heading,
            velocity=speed_ms,
            time_step=t
        )
        state_list.append(new_state)

    # Create trajectory and prediction
    if state_list:
        trajectory = Trajectory(1, state_list)
        vessel_shape = Rectangle(length=vessel_length, width=vessel_length / 4)
        prediction = TrajectoryPrediction(trajectory, vessel_shape)

        return initial_state, prediction, vessel_shape
    else:
        return initial_state, None, Rectangle(length=vessel_length, width=vessel_length / 4)


# Vessel 1: Large container ship (eastbound)
print("  - Container ship (eastbound)")
vessel1_start = np.array([-1400, -100])
vessel1_end = np.array([1400, -100])
vessel1_initial, vessel1_prediction, vessel1_shape = create_vessel_trajectory(
    vessel1_start, vessel1_end, 12, 200  # 12 knots, 200m long
)

vessel1 = DynamicObstacle(
    scenario.generate_object_id(),
    ObstacleType.MOTORVESSEL,
    vessel1_shape,
    vessel1_initial,
    vessel1_prediction
)
scenario.add_objects(vessel1)

# Vessel 2: Tanker (westbound, slower)
print("  - Tanker (westbound)")
vessel2_start = np.array([1300, 150])
vessel2_end = np.array([-1300, 150])
vessel2_initial, vessel2_prediction, vessel2_shape = create_vessel_trajectory(
    vessel2_start, vessel2_end, 8, 250  # 8 knots, 250m long
)

vessel2 = DynamicObstacle(
    scenario.generate_object_id(),
    ObstacleType.MOTORVESSEL,
    vessel2_shape,
    vessel2_initial,
    vessel2_prediction
)
scenario.add_objects(vessel2)

# Vessel 3: Fishing vessel (southbound in secondary channel)
print("  - Fishing vessel (southbound)")
vessel3_start = np.array([200, 700])
vessel3_end = np.array([200, -700])
vessel3_initial, vessel3_prediction, vessel3_shape = create_vessel_trajectory(
    vessel3_start, vessel3_end, 6, 80  # 6 knots, 80m long
)

vessel3 = DynamicObstacle(
    scenario.generate_object_id(),
    ObstacleType.MOTORVESSEL,
    vessel3_shape,
    vessel3_initial,
    vessel3_prediction
)
scenario.add_objects(vessel3)

# Vessel 4: Pilot boat (high speed, northbound in secondary channel)
print("  - Pilot boat (northbound)")
vessel4_start = np.array([300, -600])
vessel4_end = np.array([300, 600])
vessel4_initial, vessel4_prediction, vessel4_shape = create_vessel_trajectory(
    vessel4_start, vessel4_end, 20, 40  # 20 knots, 40m long
)

vessel4 = DynamicObstacle(
    scenario.generate_object_id(),
    ObstacleType.MOTORVESSEL,
    vessel4_shape,
    vessel4_initial,
    vessel4_prediction
)
scenario.add_objects(vessel4)

# Create planning problem (for ego vessel)
print("Creating planning problem...")
ego_initial_position = np.array([-200, 0])
ego_goal_position = np.array([800, 0])
ego_speed = 10 * 0.514444  # 10 knots in m/s

# Calculate heading and time to goal
ego_heading = 0.0  # Eastward
distance_to_goal = np.linalg.norm(ego_goal_position - ego_initial_position)
time_to_goal = distance_to_goal / ego_speed

ego_initial_state = CustomState(
    position=ego_initial_position,
    orientation=ego_heading,
    velocity=ego_speed,
    time_step=0
)

# Goal region with time window
goal_time_end = max(1, int((time_to_goal - 10) / dt))  # Allow 10s early
goal_time_start = min(total_horizon, int((time_to_goal + 20) / dt))  # Allow 20s late

goal_time_interval = Interval(start=goal_time_start, end=goal_time_end)
goal_position_shape = Circle(radius=100, center=ego_goal_position)
goal_orientation = AngleInterval(start=ego_heading - 0.5, end=ego_heading + 0.5)

goal_state = CustomState(
    position=goal_position_shape,
    orientation=goal_orientation,
    time_step=goal_time_interval
)

goal_region = GoalRegion([goal_state])
planning_problem = PlanningProblem(
    scenario.generate_object_id(),
    ego_initial_state,
    goal_region
)
planning_problem_set = PlanningProblemSet([planning_problem])

# Save and visualize
print("Saving scenario...")
author = 'Maritime Scenario Generator'
affiliation = 'CommonOcean Tutorial'
source = 'Generated realistic navigation scenario'
tags = {Tag.CRITICAL, Tag.HARBOUR}

fw = CommonOceanFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
filename = f'{benchmark_id}.xml'
fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

print(f"Scenario saved as: {filename}")

# Create visualization
print("Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Maritime Navigation Scenario - Time Evolution', fontsize=16)

time_frames = [0, 5, 10, 14]  # Show different time points
for i, frame in enumerate(time_frames):
    ax = axes[i // 2, i % 2]
    plt.sca(ax)

    # Draw scenario at specific time frame
    draw_object(scenario, draw_params={'time_begin': frame, 'trajectory_steps': 0})
    draw_object(planning_problem_set, draw_params={'time_begin': frame})

    ax.set_aspect('equal')
    ax.set_title(f'Time Step {frame} ({frame * dt:.0f}s)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== SCENARIO SUMMARY ===")
print(f"Waterways: 2 (main channel + secondary)")
print(f"Static obstacles: {len([obj for obj in scenario.obstacles if isinstance(obj, StaticObstacle)])}")
print(f"Dynamic vessels: {len([obj for obj in scenario.obstacles if isinstance(obj, DynamicObstacle)])}")
print(f"Navigation aids: 2 markers")
print(f"Shallow areas: 2")
print(f"Planning problems: 1 (ego vessel)")
print(f"Total simulation time: {total_horizon * dt} seconds")

# Create a detailed plot of the initial state
plt.figure(figsize=(14, 10))
draw_object(scenario, draw_params={'time_begin': 0, 'trajectory_steps': 5})
draw_object(planning_problem_set, draw_params={'time_begin': 0})
plt.gca().set_aspect('equal')
plt.title('Maritime Navigation Scenario - Initial State with Trajectories', fontsize=14)
plt.xlabel('East (m)', fontsize=12)
plt.ylabel('North (m)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add legend explaining the scenario
legend_text = """
Scenario Elements:
• Container Ship (200m): Eastbound at 12 knots
• Tanker (250m): Westbound at 8 knots  
• Fishing Vessel (80m): Southbound at 6 knots
• Pilot Boat (40m): Northbound at 20 knots
• Static obstacles: Island, platform, buoys, wreck
• Navigation aids: Port/starboard markers
• Ego vessel goal: Navigate safely eastward
"""

plt.text(-1400, -1500, legend_text, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.show()

print("\nScenario generation complete!")