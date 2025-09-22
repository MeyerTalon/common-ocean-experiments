import numpy as np
import random
from commonocean.scenario.scenario import Scenario
from commonocean.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonocean.scenario.trajectory import Trajectory
from commonocean.scenario.state import InitialState, State
from commonroad.geometry.shape import Rectangle, Circle, Polygon
from commonocean.prediction.prediction import TrajectoryPrediction
from commonocean.planning.planning_problem import PlanningProblem, GoalRegion, Goal
from commonocean.common.file_writer import CommonOceanFileWriter, OverwriteExistingFile
from commonocean.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt


def create_maritime_scenario():
    """Create a realistic maritime navigation scenario with vessels and obstacles."""

    # Scenario parameters
    dt = 5.0  # seconds
    horizon = 15  # time steps
    scenario_id = f"maritime_scenario_{random.randint(1000, 9999)}"

    # Create scenario
    scenario = Scenario(dt, scenario_id)

    # Define water area boundaries (in meters)
    water_width = 2000
    water_height = 1500

    # Create static obstacles (islands, rocks, shallow areas)
    static_obstacles = []

    # Island 1 - Large irregular island
    island1_vertices = np.array([
        [300, 800], [450, 850], [500, 750], [480, 650],
        [400, 600], [320, 650], [300, 750]
    ])
    island1_shape = Polygon(island1_vertices)
    static_obs1 = StaticObstacle(
        obstacle_id=1,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=island1_shape
    )
    static_obstacles.append(static_obs1)

    # Island 2 - Circular rock formation
    rock_shape = Circle(radius=80, center=np.array([1200, 300]))
    static_obs2 = StaticObstacle(
        obstacle_id=2,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=rock_shape
    )
    static_obstacles.append(static_obs2)

    # Shallow area - Rectangular sandbar
    sandbar_shape = Rectangle(length=300, width=100,
                              center=np.array([900, 900]),
                              orientation=np.deg2rad(30))
    static_obs3 = StaticObstacle(
        obstacle_id=3,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=sandbar_shape
    )
    static_obstacles.append(static_obs3)

    # Small rock outcrop
    rock2_vertices = np.array([
        [1500, 1100], [1550, 1120], [1540, 1050], [1490, 1070]
    ])
    rock2_shape = Polygon(rock2_vertices)
    static_obs4 = StaticObstacle(
        obstacle_id=4,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=rock2_shape
    )
    static_obstacles.append(static_obs4)

    # Add static obstacles to scenario
    for obs in static_obstacles:
        scenario.add_objects(obs)

    # Create dynamic obstacles (vessels)
    vessels = []

    # Vessel 1: Cargo ship - straight transit
    cargo_start = np.array([100, 400])
    cargo_velocity = np.array([15, 3])  # m/s (slower, large vessel)
    cargo_heading = np.arctan2(cargo_velocity[1], cargo_velocity[0])

    cargo_states = []
    for t in range(horizon):
        pos = cargo_start + cargo_velocity * t * dt
        state = State(
            position=pos,
            velocity=cargo_velocity[0],
            orientation=cargo_heading,
            time_step=t
        )
        cargo_states.append(state)

    cargo_shape = Rectangle(length=120, width=25, center=np.array([0, 0]))
    cargo_trajectory = Trajectory(initial_time_step=0, state_list=cargo_states)
    cargo_prediction = TrajectoryPrediction(trajectory=cargo_trajectory, shape=cargo_shape)

    cargo_vessel = DynamicObstacle(
        obstacle_id=10,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=cargo_shape,
        initial_state=InitialState(
            position=cargo_start,
            velocity=cargo_velocity[0],
            orientation=cargo_heading,
            time_step=0
        ),
        prediction=cargo_prediction
    )
    vessels.append(cargo_vessel)

    # Vessel 2: Ferry - curved path avoiding island
    ferry_positions = []
    ferry_start = np.array([1800, 800])

    # Create waypoints for curved trajectory
    waypoints = [
        ferry_start,
        np.array([1600, 750]),
        np.array([1400, 700]),
        np.array([1100, 650]),
        np.array([800, 600]),
        np.array([600, 500]),
        np.array([400, 400])
    ]

    # Interpolate smooth trajectory
    t_waypoints = np.linspace(0, horizon - 1, len(waypoints))
    t_interp = np.arange(horizon)
    x_interp = np.interp(t_interp, t_waypoints, [wp[0] for wp in waypoints])
    y_interp = np.interp(t_interp, t_waypoints, [wp[1] for wp in waypoints])

    ferry_states = []
    for i in range(horizon):
        pos = np.array([x_interp[i], y_interp[i]])
        if i > 0:
            velocity_vec = (pos - np.array([x_interp[i - 1], y_interp[i - 1]])) / dt
            velocity = np.linalg.norm(velocity_vec)
            orientation = np.arctan2(velocity_vec[1], velocity_vec[0])
        else:
            velocity = 12.0  # m/s
            orientation = np.arctan2(waypoints[1][1] - waypoints[0][1],
                                     waypoints[1][0] - waypoints[0][0])

        state = State(
            position=pos,
            velocity=velocity,
            orientation=orientation,
            time_step=i
        )
        ferry_states.append(state)

    ferry_shape = Rectangle(length=80, width=20, center=np.array([0, 0]))
    ferry_trajectory = Trajectory(initial_time_step=0, state_list=ferry_states)
    ferry_prediction = TrajectoryPrediction(trajectory=ferry_trajectory, shape=ferry_shape)

    ferry_vessel = DynamicObstacle(
        obstacle_id=11,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=ferry_shape,
        initial_state=InitialState(
            position=ferry_start,
            velocity=12.0,
            orientation=ferry_states[0].orientation,
            time_step=0
        ),
        prediction=ferry_prediction
    )
    vessels.append(ferry_vessel)

    # Vessel 3: Fishing boat - zigzag pattern
    fishing_start = np.array([200, 1200])
    fishing_base_velocity = 8.0  # m/s (slower small vessel)

    fishing_states = []
    for t in range(horizon):
        # Create zigzag pattern
        x = fishing_start[0] + fishing_base_velocity * t * dt
        y = fishing_start[1] + 100 * np.sin(t * np.pi / 4)
        pos = np.array([x, y])

        # Calculate velocity and orientation
        if t > 0:
            prev_pos = np.array([
                fishing_start[0] + fishing_base_velocity * (t - 1) * dt,
                fishing_start[1] + 100 * np.sin((t - 1) * np.pi / 4)
            ])
            velocity_vec = (pos - prev_pos) / dt
            velocity = np.linalg.norm(velocity_vec)
            orientation = np.arctan2(velocity_vec[1], velocity_vec[0])
        else:
            velocity = fishing_base_velocity
            orientation = 0

        state = State(
            position=pos,
            velocity=velocity,
            orientation=orientation,
            time_step=t
        )
        fishing_states.append(state)

    fishing_shape = Rectangle(length=25, width=8, center=np.array([0, 0]))
    fishing_trajectory = Trajectory(initial_time_step=0, state_list=fishing_states)
    fishing_prediction = TrajectoryPrediction(trajectory=fishing_trajectory, shape=fishing_shape)

    fishing_vessel = DynamicObstacle(
        obstacle_id=12,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=fishing_shape,
        initial_state=InitialState(
            position=fishing_start,
            velocity=fishing_base_velocity,
            orientation=0,
            time_step=0
        ),
        prediction=fishing_prediction
    )
    vessels.append(fishing_vessel)

    # Vessel 4: Yacht - maneuvering around obstacles
    yacht_start = np.array([1700, 100])
    yacht_velocity = 18.0  # m/s (fast small vessel)

    # Create path that avoids the circular rock
    yacht_waypoints = [
        yacht_start,
        np.array([1500, 200]),
        np.array([1300, 350]),  # Go around rock
        np.array([1000, 450]),
        np.array([700, 500]),
        np.array([400, 450])
    ]

    t_yacht = np.linspace(0, horizon - 1, len(yacht_waypoints))
    t_interp_yacht = np.arange(horizon)
    x_yacht = np.interp(t_interp_yacht, t_yacht, [wp[0] for wp in yacht_waypoints])
    y_yacht = np.interp(t_interp_yacht, t_yacht, [wp[1] for wp in yacht_waypoints])

    yacht_states = []
    for i in range(horizon):
        pos = np.array([x_yacht[i], y_yacht[i]])
        if i > 0:
            velocity_vec = (pos - np.array([x_yacht[i - 1], y_yacht[i - 1]])) / dt
            velocity = np.linalg.norm(velocity_vec)
            orientation = np.arctan2(velocity_vec[1], velocity_vec[0])
        else:
            velocity = yacht_velocity
            orientation = np.arctan2(yacht_waypoints[1][1] - yacht_waypoints[0][1],
                                     yacht_waypoints[1][0] - yacht_waypoints[0][0])

        state = State(
            position=pos,
            velocity=velocity,
            orientation=orientation,
            time_step=i
        )
        yacht_states.append(state)

    yacht_shape = Rectangle(length=15, width=5, center=np.array([0, 0]))
    yacht_trajectory = Trajectory(initial_time_step=0, state_list=yacht_states)
    yacht_prediction = TrajectoryPrediction(trajectory=yacht_trajectory, shape=yacht_shape)

    yacht = DynamicObstacle(
        obstacle_id=13,
        obstacle_type=ObstacleType.UNKNOWN,
        obstacle_shape=yacht_shape,
        initial_state=InitialState(
            position=yacht_start,
            velocity=yacht_velocity,
            orientation=yacht_states[0].orientation,
            time_step=0
        ),
        prediction=yacht_prediction
    )
    vessels.append(yacht)

    # Add vessels to scenario
    for vessel in vessels:
        scenario.add_objects(vessel)

    return scenario


def visualize_scenario(scenario):
    """Visualize the maritime scenario."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create renderer
    renderer = MPRenderer(figsize=(14, 10))

    # Draw scenario at different time steps
    time_steps = [0, 5, 10, 14]  # Sample time steps to show
    colors = ['blue', 'green', 'orange', 'red']

    for idx, t in enumerate(time_steps):
        renderer.draw_params.time_begin = t
        renderer.draw_params.dynamic_obstacle.draw_shape = True
        renderer.draw_params.dynamic_obstacle.vehicle_shape.opacity = 0.3 + idx * 0.2

        scenario.draw(renderer)

    renderer.render()
    plt.title(f"Maritime Navigation Scenario\nTime Step Size: 5s, Horizon: 15 steps")
    plt.xlabel("Distance [m]")
    plt.ylabel("Distance [m]")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def save_scenario(scenario, filename="maritime_scenario.xml"):
    """Save scenario to CommonOcean XML format."""
    fw = CommonOceanFileWriter(
        scenario=scenario,
        planning_problem_set=None
    )
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
    print(f"Scenario saved to {filename}")


if __name__ == "__main__":
    # Generate random seed for uniqueness
    random.seed()
    np.random.seed()

    # Create scenario
    print("Generating maritime navigation scenario...")
    scenario = create_maritime_scenario()

    # Print scenario information
    print(f"\nScenario ID: {scenario.scenario_id}")
    print(f"Time step size: {scenario.dt} seconds")
    print(f"Total horizon: 15 time steps ({15 * scenario.dt} seconds)")
    print(f"Number of static obstacles: {len(scenario.static_obstacles)}")
    print(f"Number of vessels: {len(scenario.dynamic_obstacles)}")

    # Visualize
    visualize_scenario(scenario)

    # Optionally save to file
    # save_scenario(scenario)