# -- Public Imports
import wandb

# -- Private Imports
from constants import *
from environment import ORANEnv
from agents.dqn import BaseAgentDQN

# -- Global Variables


# -- Functions

def run_drl(env, agent):
    # Logging variables
    episode_rewards = []
    episode_losses = []

    for episode in range(NUM_EPISODES):
        # Reset the environment at the start of each episode
        state = env.reset()
        episode_reward = 0
        episode_loss = 0

        for step in range(MAX_STEP):
            # Agent selects an action based on the current state
            action = agent.act(state)

            # Execute the action in the environment
            next_state, reward, done = env.step(action)

            # Store the experience in the replay buffer
            agent.record((state, action, reward, next_state))

            # Train the agent if enough samples are available in the buffer
            if agent.buffer_counter > agent.batch_size:
                loss, q_values = agent.update()
                episode_loss += loss.numpy()

            # Soft update the target network
            agent.update_target()

            # Update the current state and accumulate the episode reward
            state = next_state
            episode_reward += reward

            # Terminate the episode if the environment signals completion
            if done:
                break

        # Log the episode results
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / (step + 1))

        # Log metrics to wandb
        wandb.log({
            "Episode": episode + 1,
            "Episode Reward": episode_reward,
            "Average Loss": episode_loss / (step + 1),
            "Epsilon": agent.epsilon,
        })

        # Print training progress
        print(f"Episode {episode + 1}/{NUM_EPISODES}: "
              f"Reward = {episode_reward}, "
              f"Average Loss = {episode_loss / (step + 1):.4f}, "
              f"Epsilon = {agent.epsilon:.4f}")

        # Save the model weights periodically
        if (episode + 1) % 100 == 0:
            agent.save_model_weights()

    # Save the final model weights after training
    agent.save_model_weights()

    # Log the final episode rewards and losses to wandb
    wandb.log({
        "Final Episode Rewards": episode_rewards,
        "Final Episode Losses": episode_losses,
    })

    # Finish the wandb run
    wandb.finish()