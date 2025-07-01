from imitation_learning_lerobot.envs import DishWasherEnv

if __name__ == '__main__':
    env = DishWasherEnv(render_mode="human")
    env.run()
