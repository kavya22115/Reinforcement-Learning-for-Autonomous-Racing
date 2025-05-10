import gym
import numpy as np
import torch
from car_dqn import CarRacingDQN
from dqn import DQNAgent
from processimage import processimage

# Monkey-patch: if np.bool8 is not available, alias it to np.bool_
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
def main():
    env = gym.make("CarRacing-v2", render_mode="human")
    pic_size = (96, 96)
    num_frame_stack = 3

    model_config = {
        'gamma': 0.95,
        'lr': 1e-3,
        'batch_size': 64,
        'buffer_size': int(1e5),
        'target_update_freq': 1000,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'eps_decay': 100000
    }
    
    car_dqn = CarRacingDQN(env, max_negative_rewards=100)
    agent = DQNAgent(env, car_dqn.all_actions, pic_size=pic_size, **model_config)
    
    num_episodes = 1000
    for episode in range(num_episodes):
        obs_info = env.reset()
        state = obs_info[0] if isinstance(obs_info, tuple) else obs_info
        state = processimage.process_image(state)
        agent.memory.start_new_episode(state)
        total_reward = 0
        done = False
        
        while not done:
            # Render the current state
            env.render()  # This will open a window and update it every step
            
            state_tensor = agent.memory.current_state()
            action = agent.select_action(state_tensor)
            
            reward = 0
            for _ in range(3):  # Frame skip
                step_return = env.step(car_dqn.all_actions[action.item()].cpu().numpy().tolist())
                if len(step_return) == 4:  # Old Gym API
                    next_state, r, done, info = step_return
                else:  # New Gym API
                    next_state, r, terminated, truncated, info = step_return
                    done = terminated or truncated
                
                reward += r
                if done:
                    break
            
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            next_state = processimage.process_image(next_state)
            agent.memory.add_experience(next_state, action.item(), done, reward)
            agent.update_model()
            
            total_reward += reward
            agent.steps_done += 1
            
            if done:
                break
        
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {agent.steps_done}")
    
    # After training, you might want to close the window properly:
    env.close()

if __name__ == "__main__":
    main()