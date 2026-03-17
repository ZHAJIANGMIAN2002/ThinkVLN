"""Trajectory generation variant that also records agent positions."""

from __future__ import annotations

import argparse
import json
import os

import habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import FogOfWarConfig, TopDownMapMeasurementConfig
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config


class StreamVLNWaypointRunner:
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None):
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.config = get_habitat_config(self.config_path)

        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    'top_down_map': TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                    )
                }
            )
            if self.data_path is not None:
                self.config.habitat.dataset.update({'data_path': self.data_path})

    def generate(self):
        os.makedirs(self.output_path, exist_ok=True)
        env = habitat.Env(config=self.config)
        summary_file = os.path.join(self.output_path, 'summary_waypoint.json')

        for episode in env.episodes:
            env.current_episode = episode
            observation = env.reset()
            follower = ShortestPathFollower(sim=env.sim, goal_radius=0.5, return_one_hot=False)

            actions = [-1]
            positions = []
            ref_path = episode.reference_path
            next_waypoint_id = 1
            vis_frames = []

            st = env.sim.get_agent_state().position
            positions.append([float(st[0]), float(st[2])])

            while not env.episode_over:
                info = env.get_metrics()
                if info.get('top_down_map') is not None:
                    vis_frames.append(observations_to_image({'rgb': observation['rgb']}, info))

                next_action = follower.get_next_action(ref_path[next_waypoint_id])
                force_end = False
                while next_action == 0:
                    next_waypoint_id += 1
                    if next_waypoint_id >= len(ref_path):
                        force_end = True
                        break
                    next_action = follower.get_next_action(ref_path[next_waypoint_id])
                if force_end:
                    break

                observation = env.step(next_action)
                actions.append(int(next_action))
                pos = env.sim.get_agent_state().position
                positions.append([float(pos[0]), float(pos[2])])

            if len(actions) != len(positions):
                n = min(len(actions), len(positions))
                actions = actions[:n]
                positions = positions[:n]

            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            result = {
                'id': episode_id,
                'key': f'{scene_id}_{episode_id}',
                'video': os.path.join('images', f'{scene_id}_{self.dataset}_{episode_id:06d}'),
                'instructions': [episode.instruction.instruction_text],
                'actions': actions,
                'positions': positions,
                'trajectory_id': episode.trajectory_id,
                'scene_id': scene_id,
                'num_frames': len(actions),
            }

            if vis_frames:
                video_output_path = os.path.join(self.output_path, 'images', f'{scene_id}_{self.dataset}_{episode_id:06d}')
                os.makedirs(video_output_path, exist_ok=True)
                images_to_video(vis_frames, video_output_path, 'trajectory', fps=6, quality=9)

            with open(summary_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory summary with waypoint positions')
    parser.add_argument('--dataset', type=str, default='r2r')
    parser.add_argument('--config_path', type=str, default='./config/vln_r2r.yaml')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    runner = StreamVLNWaypointRunner(args.dataset, args.config_path, args.output_path, args.data_path)
    runner.generate()


if __name__ == '__main__':
    main()
