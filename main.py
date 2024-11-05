from ai_agent import AIAgent
from env import env_red
from utils import generate_timestamped_id
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


def run_ai_mode(environment, episode_id=None, previous_episode_id=None, episode_length=1000):
    ai_agent = AIAgent()

    # Load previous episode's checkpoint if exists
    if previous_episode_id:
        checkpoint = f"checkpoints/agent_state_{previous_episode_id}.pkl"
        if os.path.exists(checkpoint):
            ai_agent.load_state(checkpoint)

    # Generate new episode ID if none provided
    if episode_id is None:
        episode_id = generate_timestamped_id()

    state = environment.reset()
    step = 0
    while step < episode_length:
        step += 1
        if step % 100 == 0:  # Save checkpoint less frequently
            ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")

        action = ai_agent.select_action(state)
        next_state, reward, done, _ = environment.step(action, False)

        # debug logs
        if step % 1000 == 0 or step == episode_length - 1:  # Every 1000 steps or at episode end
            current_pos = next_state["position"]
            target_pos = (309, 99)  # Target coordinates
            distance = ((current_pos[0] - target_pos[0])**2 + 
                       (current_pos[1] - target_pos[1])**2)**0.5
            
            print(f"\nEpisode step {step}/{episode_length}:")
            print(f"Total reward: {environment.total_reward:.2f}")
            print(f"Distance to target: {distance:.2f}")
            print(f"Current position: {current_pos}")

        state = next_state
        if done:
            break

    # Final save at episode end
    ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")
    environment.save_episode_stats(episode_id)
    return episode_id


def run_manual_mode(env_red):
    environment = env_red
    environment.reset()
    done = False
    while not done:
        next_state, reward, done, _ = environment.step(manual=True)


def main():
    args = parse_arguments()
    environment = env_red()

    try:
        if args.manual:
            run_manual_mode(environment)
        else:
            # change to None to start with blank q table
            initial_q_state = None

            previous_id = initial_q_state
            for episode in range(args.episodes):
                print(f"\nStarting episode {episode + 1}/{args.episodes}")
                episode_id = generate_timestamped_id()
                previous_id = run_ai_mode(
                    environment,
                    episode_id=episode_id,
                    previous_episode_id=previous_id,
                    episode_length=args.episode_length,
                )
    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
