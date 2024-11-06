from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import env_red
from utils import generate_timestamped_id
from actions import Actions
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Pokémon Red with AI or manual control."
    )
    parser.add_argument(
        "--manual", action="store_true", help="Enable manual control mode."
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--episode_length", type=int, default=300, help="Steps per episode"
    )
    return parser.parse_args()

def run_ai_mode(environment, model, episode_id=None, episode_length=1000):
    # Generate new episode ID if none provided
    if episode_id is None:
        episode_id = generate_timestamped_id()

    obs = environment.reset()
    step = 0
    while step < episode_length:
        step += 1
        action, _states = model.predict(obs, deterministic=True)  # action is already an integer

        # Pass the integer action directly to the environment
        obs, reward, done, _ = environment.step(action)

        if done:
            break

    # Save model at the end of each episode
    model.save(f"checkpoints/ppo_agent_{episode_id}.zip")
    environment.save_episode_stats(episode_id)
    return episode_id

# def run_ai_mode(environment, model, episode_id=None, episode_length=1000):
#     # Generate new episode ID if none provided
#     if episode_id is None:
#         episode_id = generate_timestamped_id()

#     obs = environment.reset()
#     step = 0
#     while step < episode_length:
#         step += 1
#         # Run environment with integer-based Actions
#         action = Actions(model.predict(obs, deterministic=True)[0])  # Convert to Actions enum
#         obs, reward, done, _ = environment.step(action)


#         if done:
#             break

#     # Save model at the end of each episode
#     model.save(f"checkpoints/ppo_agent_{episode_id}.zip")
#     environment.save_episode_stats(episode_id)
#     return episode_id


def run_manual_mode():
    environment = env_red()
    environment.reset()
    done = False
    while not done:
        next_state, reward, done, _ = environment.step(manual=True)


def main():
    args = parse_arguments()
    environment = DummyVecEnv([lambda: env_red()])

    try:
        if args.manual:
            run_manual_mode()
        else:
            # Initialize PPO model
            model = PPO("MultiInputPolicy", environment, verbose=1)

            for episode in range(args.episodes):
                print(f"\nStarting episode {episode + 1}/{args.episodes}")
                episode_id = generate_timestamped_id()
                run_ai_mode(environment, model, episode_id=episode_id, episode_length=args.episode_length)

    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
