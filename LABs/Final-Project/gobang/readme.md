## Experiment Overview

In this experiment, we are enhancing the basic Q-Learning approach (you learned from the warmup experiments) to develop a Gobang agent capable of playing on a more complex board with **size 𝑛=12 and a bound of 5**. However, as the board size increases, the state and action spaces grow rapidly, rendering the traditional Q-Learning method inadequate. To address this, we are introducing deep learning techniques to achieve a degree of generalization on unknown states.

### Key Modifications

The main modifications we're considering revolve around addressing two key challenges: ensuring legality of black moves and efficiently generating white responses. Instead of training against pure noise as an opponent, we opt for a more realistic approach, introducing the basic principles of game theory at the same time.

1. **Obtain Legal Actions: Constraining the Output of 𝑀(𝑠):**
   To ensure that black moves are always legal, we need to constrain the output of the deep learning model 𝑀(𝑠), which generates the probability distribution over the entire action space. One way to achieve this is by incorporating knowledge of legal moves into the model architecture or by post-processing the output probabilities to exclude illegal moves. By doing so, we guarantee that the black move selected is always valid within the game's rules.

2. **Generating White Responses Using 𝑀(𝑠):**
   Once we have a model 𝑀 that outputs the probability distribution over black moves given a state 𝑠, we can leverage the same model to generate white responses efficiently. This means that a single deep learning model is responsible for both selecting black actions and responding with white actions. By sharing the same model parameters, we can ensure consistency and streamline the training process.

### Training Techniques

- **Naïve Self Play:**
  Naïve Self Play is a technique where an agent plays against itself during training without any prior knowledge. The basic idea is to simulate games between two instances of the same agent, allowing it to learn from its own experiences. This approach is straightforward and easy to implement, making it suitable for training deep learning models in reinforcement learning tasks like Gobang. However, it may suffer from convergence issues or lack of diversity in gameplay strategies, which can be addressed through more sophisticated self-play algorithms.

- **Actor-Critic Architecture:**
  To train our model effectively in the defined Zero-Sum Markov Games framework, we adopt the Actor-Critic architecture. This architecture consists of two components: the actor, which selects actions based on the current policy, and the critic, which evaluates the chosen actions' goodness. By combining these two components, we aim to improve the model's policy while simultaneously estimating the state value function, facilitating more efficient and stable learning.

## Conclusion

Our experiment presents a comprehensive exploration towards developing a Gobang agent adept at navigating larger boards through the integration of deep learning techniques and reinforcement learning principles. While our efforts have been focused on addressing challenges such as legality constraints and efficient response generation, we acknowledge the possibility of shortcomings and limitations within the algorithm. There remains room for improvement and alternative approaches that could further enhance the bot's performance and adaptability. We aspire to foster an environment of continuous learning and evolution, recognizing that the pursuit of excellence is an ongoing journey.

## Running Experiments

To run the experiments defined in the configuration:

1. Activate the environment: `source .venv/bin/activate`
2. Use the unified experiment conductor:
   ```bash
   # Run all experiments defined in the config
   python -m tests.conduct_experiment --config experiments/experiment_config.yaml

   # Run a specific experiment
   python -m tests.conduct_experiment --config experiments/experiment_config.yaml --experiment baseline

   # Specify custom results directory
   python -m tests.conduct_experiment --config experiments/experiment_config.yaml --results-dir my_results
   ```
