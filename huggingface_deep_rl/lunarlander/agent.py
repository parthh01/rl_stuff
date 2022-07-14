import gym 




def main():
    env = gym.make('LunarLander-v2')
    s = env.reset()
    for t in range(1000):
        action = env.action_space.sample() 
        s,r,terminated,info = env.step(action)
        print(s.shape,r,env.action_space)
        if terminated: 
            break 
        env.render()
    
    print('done')


if __name__ == "__main__":
    main()