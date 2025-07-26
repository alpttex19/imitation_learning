from imitation_learning_lerobot.envs import TransferCubeEnv

if __name__ == '__main__':
    env = TransferCubeEnv(render_mode="human")
    env.reset()

    while True:
        env.step(None)

        env.render()
