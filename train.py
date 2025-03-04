# -- Public Imports
import wandb
import numpy as np
import tensorflow as tf

# -- Private Imports
from utils import *
from constants import *
from environment import UnifiedEnv
from agents.dqn import BaseAgentDQN

# -- Global Variables
tf.get_logger().setLevel('ERROR')

# -- Functions

def run_drl(args):
    # Set seeds
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    # Initialize wandb
    wandb.init(
        project="DRL-ORAN_Energy",
        name="DRL for Energy Saving in ORAN",
        config=args
    )

    # Initialize env and drl agent
    env = UnifiedEnv(env_type=args.env, config=None)
    agent = BaseAgentDQN(args)

    # Logging variables
    ep_rewards = []
    step_rewards = []
    avg_rewards = []
    ep_losses = []
    step_losses = []

    for episode in range(args.num_episodes):
        # Reset the environment at the start of each episode
        state = env.reset()
        episode_reward = 0
        episode_loss = 0

        for step in range(args.max_step):
            # Agent selects an action based on the current state
            action, action_idx = agent.act(state)

            # Execute the action in the environment
            next_state, reward, done = env.step(action)
            step_rewards.append(reward)

            # Store the experience in the replay buffer
            agent.record((state, action_idx, reward, next_state))

            # Train behaviour and target models
            loss = agent.update()
            loss = loss.numpy()
            episode_loss += loss
            step_losses.append(loss)
            agent.update_target()

            # Update the current state and accumulate the episode reward
            state = next_state
            episode_reward += reward

            # Log step info to wandb
            wandb.log({
                "Episode": episode+1,
                "Step": step+1,
                "Step Reward": reward,
                "Step Loss": loss,
            })

            # Terminate the episode if the environment signals completion
            if done:
                break

        # Log the episode results
        ep_rewards.append(episode_reward)
        avg_reward = np.mean(step_rewards[-args.max_step:])
        avg_rewards.append(avg_reward)

        ep_losses.append(episode_loss / (step + 1))

        # Log metrics to wandb
        wandb.log({
            "Episode": episode + 1,
            "Episode Reward": episode_reward,
            "Average Reward": avg_reward,
            "Average Loss": episode_loss / (step + 1),
        })

        # Print training progress
        print(f"Episode {episode + 1}/{args.num_episodes}: "
              f"Episode Reward = {episode_reward:.2e}, "
              f"Avg Reward = {avg_reward:.2e}, "
              f"Average Loss = {episode_loss / (step + 1):.2e}, "
              f"Action = {action_idx}, ")

        # Save the model weights periodically
        if (episode + 1) % 10 == 0:
            agent.save_model()

    # Save the final model weights after training
    agent.save_model()

    # Log the final episode rewards and losses to wandb
    wandb.log({
        "Final Episode Rewards": ep_rewards,
        "Final Step Rewards": step_rewards,
        "Final Avg Rewards": avg_rewards,
        "Final Episode Losses": ep_losses,
        "Final Step Losses": step_losses,
    })

    saves_lists(f"./lists/{args.agent}", ep_rewards, step_rewards, avg_rewards, ep_losses, step_losses)

    # Finish the wandb run
    wandb.finish()