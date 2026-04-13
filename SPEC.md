# Specification

## Mission

The goal of this project is to create an environment-first framework for reinforcement learning from verifiable rewards (RLVR) using game engines as executable verifiers.
An LLM agent will interact with a stateful game environment over multiple turns, and the game engine will verify transitions, legality, terminal outcomes, and rewards based on the agent's actions.
This framework should make it possible to train and evaluate LLM agents on reasoning tasks where correct behavior must be demonstrated through interaction, not only through a final static answer.

- The environment is the core abstraction, not a static question/answer dataset.
- Trajectories are first-class: observations, actions, rewards, and terminal states should be recorded and verifiable.
- The interaction between the agent and the game will be multi-modal, involving both text and images.
- The game engine will be the source of truth for symbolic state, legal actions, transitions, and rewards.
- Rewards may be sparse or dense, but they must be grounded in executable game logic.
- Rendered text and images are views over canonical symbolic state, not the authoritative state themselves.
- The framework should be flexible and extensible, allowing new games, reward structures, scenario generators, and renderers to be added without changing the core environment interface.

The core loop should look conceptually like this:

```python
observation = env.reset(task_config)
done = False

while not done:
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
```

Each environment should define:

- a canonical game state
- an observation format for the model
- an action format that can be parsed and verified
- a transition function backed by game rules
- a reward function backed by executable verification
- terminal conditions and trajectory metadata

## Games

Implemented games:

- Chess
- 2048
- Connect 4

Possible next additions:

- Checkers
- Go


## Rewards

- We want to support both sparse and dense rewards.
- 
