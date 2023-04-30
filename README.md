Download Link: https://assignmentchef.com/product/solved-homework-2-value-iteration-policy-iteration-and-q-learning-cmu-10-703
<br>
<h1>Problem 1: Basics &amp; MDPs (27 pts)</h1>

For this problem, assume the MDP has finite state and action spaces. Let <em>V <sup>π</sup></em>(<em>s</em>) be the value of state <em>s </em>under a policy <em>π</em>, which is the expected return when starting in state <em>s </em>at time <em>t </em>and following the policy <em>π </em>thereafter:

<ol>

 <li>(6 pts) Assume a discount factor less than one: <em>γ &lt; </em> Show that for all policies <em>π </em>and states <em>s </em>∈ <em>S</em>, the value function <em>V <sup>π</sup></em>(<em>s</em>) is well-defined, <em>i.e.</em>. That is, show that (<em>i</em>) the infinite sum converges even if the episode never terminates; and (<em>ii</em>) the outer expectation E<em>π</em>[·] is bounded. Hint: .</li>

 <li>(6 pts) For each of the following three statements, answer “True” or “False.” In addition, explain why in 1-2 sentences. As a reminder, we are considering MDPs with finite state and action spaces.

  <ul>

   <li>For every MDP, there exists a unique optimal policy.</li>

   <li>If an MDP has multiple optimal policies, then each has the same value function.</li>

   <li>There are some fully-observed MDPs on which stochastic policies can obtain higher reward than deterministic policies.</li>

  </ul></li>

 <li>(9 pts) Consider three MDPs with two states ({<em>s</em><sub>1</sub><em>,s</em><sub>2</sub>}) and two actions ({<em>a</em><sub>1</sub><em>,a</em><sub>2</sub>}).</li>

</ol>

MDP 1:

Transition function:

<ul>

 <li><em>P</em>(<em>s</em><sub>1</sub>|<em>a</em><sub>1</sub><em>,s</em><sub>1</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>2</sub>|<em>a</em><sub>2</sub><em>,s</em><sub>1</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>2</sub>|<em>a</em><sub>1</sub><em>,s</em><sub>2</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>2</sub>|<em>a</em><sub>2</sub><em>,s</em><sub>2</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sup>0</sup>|<em>a,s</em>) = 0 for all other <em>s,a,s</em><sup>0 </sup>triples</li>

</ul>

Reward function:

<h2>• R(s<sub>1</sub>,a<sub>2</sub>,s<sub>2</sub>) = 1</h2>

<ul>

 <li><em>R</em>(<em>s,a,s</em><sup>0</sup>) = 0 for all other <em>s,a,s</em><sup>0 </sup>triples</li>

</ul>

MDP 2:

Transition function is the same as MDP 1.

Reward function:

<h2>• R(s<sub>2</sub>,a<sub>1</sub>,s<sub>2</sub>) = 1 • R(s<sub>2</sub>,a<sub>2</sub>,s<sub>2</sub>) = 1</h2>

<ul>

 <li><em>R</em>(<em>s,a,s</em><sup>0</sup>) = 0 for all other <em>s,a,s</em><sup>0 </sup>triples</li>

</ul>

MDP 3:

Transition function:

<ul>

 <li><em>P</em>(<em>s</em><sub>1</sub>|<em>a</em><sub>1</sub><em>,s</em><sub>1</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>2</sub>|<em>a</em><sub>2</sub><em>,s</em><sub>1</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>1</sub>|<em>a</em><sub>1</sub><em>,s</em><sub>2</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sub>2</sub>|<em>a</em><sub>2</sub><em>,s</em><sub>2</sub>) = 1</li>

 <li><em>P</em>(<em>s</em><sup>0</sup>|<em>a,s</em>) = 0 for all other <em>s,a,s</em><sup>0 </sup>triples Reward function:</li>

</ul>

<h2>• R(s<sub>1</sub>,a<sub>1</sub>,s<sub>1</sub>) = 1 • R(s<sub>2</sub>,a<sub>2</sub>,s<sub>2</sub>) = 1</h2>

<ul>

 <li><em>R</em>(<em>s,a,s</em><sup>0</sup>) = 0 for all other <em>s,a,s</em><sup>0 </sup>triples</li>

 <li>(6 pts) Assume a discount factor <em>γ </em>= 1. For each of three MDPs above, does there exists a policy whose value function is infinite for <em>s</em><sub>1</sub>. If the answer is yes, please describe the set of all such policies; otherwise, please describe the set of all policies whose value function is maximum for <em>s</em><sub>1 </sub>across all policies.</li>

 <li>(3 pts) Now assume a discount factor of 1 , where is some small positive value. For each of three MDPs above, describe the set of optimal policies.</li>

</ul>

<ol start="4">

 <li>(6 pts) The discount factor <em>γ </em>can change the optimal policy. Describe an MDP and two policies, each of which is optimal for a different value of <em>γ</em>.</li>

</ol>

<h1>Problem 2: Value Iteration &amp; Policy Iteration (26+4)</h1>

In this problem, you will implement value iteration and policy iteration. Throughout this problem, initialize the value functions as zero for all states and break ties in order of state numbering (as described further below).

We will be working with a different version of the OpenAI Gym environment

Deterministic-*-FrozenLake-v0<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>, defined in deeprl hw2q2/lake envs.py. You can check README.md for specific coding instructions. Starter code is provided in deeprl hw2q2/rl.py, with useful HELPER functions at the end of the file! You may use either Python 2.7 or 3. We have provided two different maps, a 4 × 4 map and a 8 × 8 map:

FFFFFSFF

FFFFFFFF

<table width="236">

 <tbody>

  <tr>

   <td width="125">FHSFFGHFFHHFFFFF</td>

   <td width="111">HHHHHHFFFFFFFFFFFFFFFFFFFHFFFHHF</td>

  </tr>

 </tbody>

</table>

FHFFHFHH

FGFFFFFF

There are four different tile types: Start (S), Frozen (F), Hole (H), and Goal (G).

<ul>

 <li>The agent starts in the Start tile at the beginning of each episode.</li>

 <li>When the agent lands on a Frozen or Start tile, it receives 0 reward.</li>

 <li>When the agent lands on a Hole tile, it receives 0 reward and the episode ends.</li>

 <li>When the agent lands on the Goal tile, it receives +1 reward and the episode ends.</li>

</ul>

States are represented as integers numbered from left to right, top to bottom starting at zero. For example in a 4 × 4 map, the upper-left corner is state 0 and the bottom-right corner is state 15:

0       1       2       3

4       5       6       7

8       9      10    11

12    13     14    15

<strong>Note: </strong>Be careful when implementing value iteration and policy evaluation. Keep in mind that in this environment, the reward function depends on the current state, the current action, and the <strong>next state</strong>. Also, terminal states are slightly different. Think about the backup diagram for terminal states and how that will affect the Bellman equation.

In this section, we will use the deterministic versions of the FrozenLake environment. Answer the following questions for the maps Deterministic-4×4-FrozenLake-v0 and Deterministic-8×8-FrozenLake-v0.

<ol>

 <li>(4 pts) For each domain, find the optimal policy using <strong>synchronous policy iteration </strong>(see Fig. 3). Specifically, you will implement policy iteration sync() in deeprl hw2q2/rl.py, writing the policy evaluation steps in evaluate policy sync() and policy improvement steps in improve policy(). Record (1) the number of policy improvement steps and (2) the total number of policy evaluation steps. Use a discount factor of <em>γ </em>= 0<em>.</em> Use a stopping tolerance of <em>θ </em>= 10<sup>−3 </sup>for the policy evaluation step.</li>

</ol>

<table width="356">

 <tbody>

  <tr>

   <td width="137"><strong>Environment</strong></td>

   <td width="120"><strong># Policy</strong><strong>Improvement</strong><strong>Steps</strong></td>

   <td width="99"><strong>Total #</strong><strong>Policy</strong><strong>Evaluation Steps</strong></td>

  </tr>

  <tr>

   <td width="137">Deterministic-4×4</td>

   <td width="120"> </td>

   <td width="99"> </td>

  </tr>

  <tr>

   <td width="137">Deterministic-8×8</td>

   <td width="120"> </td>

   <td width="99"> </td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>(2 pts) Show the optimal policy for the Deterministic-4×4 and 8×8 maps as grids of letters with “U”, “D”, “L”, “R” representing the actions up, down, left, right respectively.</li>

</ol>

See Figure 1 for an example of the 4×4 map. Helper: display policy letters().

<ol start="3">

 <li>(2 pts) Find the value functions of the policies for these two domains. Plot each as a color image, where each square shows its value as a color. See Figure 2 for an example for the 4×4 domain. Helper function: value func heatmap().</li>

</ol>

−0.5                                                                                                                                                                                                                       0.0

Figure 1: An example (deterministic) policy Figure 2: Example of value function color for a 4 × 4 map of the FrozenLake-v0 envi- plot for a 4 × 4 map of the FrozenLake-v0 ronment. L, D, U, R represent the actions environment. Make sure you include the up, down, left, right respectively. color bar or some kind of key.

Figure 3: Synchronous policy iteration, taken from Section 4.3 of Sutton &amp; Barto’s RL book (2018).

Figure 4: Synchronous value iteration, taken from Section 4.4 of Sutton &amp; Barto’s RL book (2018).

<ul>

 <li>Notes on <strong>Asynchronous v.s. Synchronous </strong>(value iteration &amp; polocy iteration): The main difference between sync and async versions is that: whether all the updates are performed in-place (async) or not (sync). Take the synchronous value iteration for example, at some time step <em>k</em>, you would maintain two separate vectors <em>V<sub>k </sub></em>and <em>V<sub>k</sub></em><sub>+1 </sub>and perform updates of the following form inside the loop as described in Figure 4: <em>V<sub>k</sub></em><sub>+1</sub>(<em>s</em>) ←− max<sup>X</sup><em>p</em>(<em>s</em><sup>0</sup><em>,r</em>|<em>s,a</em>)[<em>r </em>+ <em>γV<sub>k</sub></em>(<em>s</em><sup>0</sup>)]<em>.</em></li>

</ul>

<em>a</em>

<em>s</em><sup>0</sup><em>,r</em>

For asynchronous updates, you don’t need two copies of the vector as above.

<ol start="4">

 <li>(4 pts) For both domains, find the optimal value function directly using <strong>synchronous value iteration </strong>(see Fig. 4). Specifically, you will implement value iteration sync()</li>

 <li>(2 pts) Plot these two value functions as color images, where each square shows its value as a color. See Figure 2 for an example for the 4×4 domain.</li>

 <li>(2 pts) Convert both optimal value functions to the optimal policies. Show each policy as a grid of letters with “U”, “D”, “L”, “R” representing the actions up, down, left, right respectively. See Figure 1 for an example of the expected output for the 4×4 domain.</li>

 <li>(2 pts) For both of the two domains, measure the average run-time for your two algorithms (do not include as part of your answer). Which algorithm was faster in your case (policy iteration or value iteration)? Which would you expect to be faster? Would you generally expect any differences in the value function?</li>

 <li>(4 pts) Implement <strong>asynchronous policy iteration </strong>using two heuristics:

  <ul>

   <li>The first heuristic is to sweep through the states in the order they are defined in the gym environment. Specifically, you will implement</li>

  </ul></li>

</ol>

policy iteration async ordered() in deeprl hw2q2/rl.py, writing the policy evaluation step in evaluate policy async ordered().

<ul>

 <li>The second heuristic is to choose a random permutation of the states at each iteration and sweep through all of them. Specifically, you will implement policy iteration async randperm() in deeprl hw2q2/rl.py, writing the policy evaluation step in evaluate policy async randperm().</li>

</ul>

Fill in the table below with the results for Deterministic-8×8-FrozenLake-v0. Run one trial for the <strong>first </strong>(“async ordered”) heuristic. Run <strong>ten trials </strong>for the <strong>second </strong>(“async randperm”) heuristic and report the <strong>average</strong>. Use <em>γ </em>= 0<em>.</em>9. Use a stopping tolerance of 10<sup>−3</sup>.

<ol start="9">

 <li>(4 pts) Implement <strong>asynchronous value iteration </strong>using two heuristics:

  <ul>

   <li>The first heuristic is to sweep through the states in the order they are defined in the gym environment. Specifically, you will implement value iteration async ordered() in deeprl hw2q2/rl.py.</li>

   <li>The second heuristic is to choose a random permutation of the states at each iteration and sweep through all of them. Specifically, you will implement value iteration async randperm() in deeprl hw2q2/rl.py.</li>

  </ul></li>

</ol>

Fill in the table below with the results for Deterministic-8×8-FrozenLake-v0. Run one trial for the <strong>first </strong>(“async ordered”) heuristic. Run <strong>ten trials </strong>for the <strong>second </strong>(“async randperm”) heuristic and report the <strong>average</strong>. Use <em>γ </em>= 0<em>.</em>9. Use a stopping tolerance of 10<sup>−3</sup>.

<ol start="10">

 <li>(<strong>4 pts, ptional</strong>) Now, you can use a domain-specific heuristic for asynchronous value iteration (value iteration async custom()) to beat the heuristics defined in Q2.9. Specifically, you will sweep through the entire state space ordered by Manhattan distance to goal.

  <ul>

   <li>Fill in the table below (use a stopping tolerance of 10<sup>−3</sup>).</li>

   <li>In what cases would you expect this “goal distance” heuristic to perform best?</li>

  </ul></li>

</ol>

Briefly explain why.

<h1>Problem 3: DQN (47+20 pts)</h1>

In this problem you will implement Q-learning, using tabular and learned representations for the Q-function. This question will be graded out of 47 points, but you can earn up to 67 points by completing the extra credit problem (3.3c).

<h2><strong>Problem 3.1: Relations among Q &amp; V &amp; C (13 pts)</strong></h2>

The objective of this question is to understand different Bellman equations, their strengths and limitations. Consider the Bellman Equation for Value function,

<em> .</em>

If we continue expanding the value function <em>V </em>(<em>s</em><sub>2</sub>) using its own Bellman equation, then we obtain a repeating structure:

<em> .</em>

There are a few more ways in which we can group this repeating sequence. First, we can capture the sequence starting at <em>R</em>(<em>s,a</em>) and ending at max, and observe that it too has a repeating substructure property:

<em>.</em>

We’ll call this repeating expression the state-value function <em>Q</em>(<em>s,a</em>) and use it to rewrite the Bellman equation as:

<em>Q</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) = <em>R</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) + <em>γ </em><sup>X</sup><em>T</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,s</em><sub>2</sub>)max<em>Q</em>(<em>s</em><sub>2</sub><em>,a</em><sub>2</sub>)<em>.</em>

<em>a</em>2 <em>s</em>2

Next, we can capture another pattern by grouping the expression beginning at <em>γ </em>and ending at <em>R</em>(<em>s,a</em>):

<em> .</em>

We’ll call this repeating expression the continuation function <em>C</em>(<em>s,a</em>), which can be written in terms of the value function:

<em>C</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) = <em>γ </em><sup>X</sup><em>T</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,s</em><sub>2</sub>)<em>V </em>(<em>s</em><sub>2</sub>)<em>.</em>

<em>s</em>2

<ol>

 <li>(3 pts) Derive the recurrence relation (Bellman equation) for <em>C</em>(<em>s,a</em>).</li>

 <li>(4 pts) Fill the following table to express the three functions in terms of each other.</li>

</ol>

<table width="559">

 <tbody>

  <tr>

   <td width="59"> </td>

   <td width="224">V(s)</td>

   <td width="154">Q(s,a)</td>

   <td width="122">C(s,a)</td>

  </tr>

  <tr>

   <td width="59">V(s)</td>

   <td width="224">V(s) = V(s)</td>

   <td width="154">V(s) = max<em><sub>a </sub>Q</em>(<em>s,a</em>)</td>

   <td width="122">(<em>a</em>)</td>

  </tr>

  <tr>

   <td width="59">Q(s,a)</td>

   <td width="224">(<em>b</em>)</td>

   <td width="154">Q(s,a) = Q(s,a)</td>

   <td width="122">(<em>c</em>)</td>

  </tr>

  <tr>

   <td width="59">C(s,a)</td>

   <td width="224"><em>C</em>(<em>s,a</em>) = <em>γ </em><sup>P</sup><em><sub>s</sub></em>0 <em>T</em>(<em>s,a,s</em><sup>0</sup>)<em>V </em>(<em>s</em><sup>0</sup>)</td>

   <td width="154">(<em>d</em>)</td>

   <td width="122">C(s,a) = C(s,a)</td>

  </tr>

 </tbody>

</table>

Use the relation between the functions and your understanding of MDPs to answer the following True/False questions. Please include a 1-2 sentence explanation for each. Consider the scenario when we want to compute the optimal action without the knowledge of transition function <em>T</em>(<em>s,a,s</em><sup>0</sup>).

<ol start="3">

 <li>(2 pts) Can you derive the optimal policy given only <em>Q</em>(<em>s,a</em>)?</li>

 <li>(2 pts) Can you derive the optimal policy given only <em>V </em>(<em>s</em>) and <em>R</em>(<em>s,a</em>)?</li>

 <li>(2 pts) Can you derive the optimal policy given only <em>C</em>(<em>s,a</em>) and <em>R</em>(<em>s,a</em>)?</li>

</ol>

<h2><strong>Problem 3.2: Temporal Difference &amp; Monte Carlo (4 pts)</strong></h2>

Answer the true/false questions below, providing one or two sentences for <strong>explanation</strong>.

<ol>

 <li>(2 pts) TD methods can’t learn in an online manner since they require full trajectories.</li>

 <li>(2 pts) MC can be applied even with non-terminating episodes.</li>

</ol>

<h2><strong>Problem 3.3: DQN Implementation (30 + 20 pts)</strong></h2>

You will implement DQN and use it to solve two problems in OpenAI Gym: Cartpole-v0 and MountainCar-v0. While there are many (fantastic) implementations of DQN on Github, the goal of this question is for you to implement DQN from scratch <em>without </em>looking up code online.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> Please write your code in the DQN implementation.py. You are free to change/delete the template code if you want.

<strong>Code Submission</strong>: Your code should be reasonably well-commented in key places of your implementation. Make sure your code also has a README file.

<strong>How to measure if I ”solved” the environment? </strong>You should achieve the reward of 200 (Cartpole-v0) and around -110 or higher (MountainCar-v0) in consecutive 50 trials. <em>i.e. </em>evaluate your policy on 50 episodes.

<strong>Runtime Estimation</strong>: To help you better manage your schedule, we provide you with reference runtime of DQN on MacBook Pro 2018. For Cartploe-v0, it takes 5 minuts to first reach a reward of 200 and 68 minutes to finish 5000 episodes. For MountainCar-v0, it takes 40 ∼ 50 minutes to reach a reward around -110 and 200 minutes to finish 10000 episodes.

<ul>

 <li><strong>(30 pts) </strong>Implement a deep Q-network with experience replay. While the original DQN paper [3] uses a convolutional architecture, a neural network with 3 fully-connected layers should suffice for the low-dimensional environments that we are working with. For the deep Q-network, look at the QNetwork and DQN Agent classes in the code. You will have to implement the following:

  <ul>

   <li>Create an instance of the Q Network class.</li>

   <li>Create a function that constructs a greedy policy and an exploration policy (greedy) from the Q values predicted by the Q Network.</li>

   <li>Create a function to train the Q Network, by interacting with the environment.</li>

   <li>Create a function to test the Q Network’s performance on the environment.</li>

  </ul></li>

</ul>

For the replay buffer, you should use the experimental setup of [3] to the extent possible. Starting from the Replay Memory class, implement the following functions:

<ul>

 <li>Append a new transition from the memory.</li>

 <li>Sample a batch of transitions from the memory to train your network.</li>

 <li>Collect an initial number of transitions using a random policy.</li>

 <li>Modify your training function of your network to learn from experience sampled <em>from the memory</em>, rather than learning online from the agent.</li>

</ul>

Train your network on both the CartPole-v0 environment and the MountainCar-v0 environment (separately) until convergence, <em>i.e. </em>train a different network for each environment. We recommend that you periodically checkpoint your network to ensure no work is lost if your program crashes. Answer following questions in your report: (a) (20 pts) Describe your implementation, including the optimizer, the neural network architecture and any hyperparameters you used.

<ul>

 <li>(5 pts) For each environment, plot the average cumulative test reward throughout training.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> You are required to plot at least 2000 more episodes after you solve CartPole-v0, and at least 1000 more episodes after you solve MountainCar-v0. To do this, every 100 episodes, evaluate the current policy for 20 episodes and average the total reward achieved. Note that in this case we are interested in total reward without discounting or truncation.</li>

 <li>(5 pts) For each environment, plot the TD error throughout training. Does the TD error decrease when the reward increases? Suggest a reason why this may or may not be the case.</li>

 <li>We want you to generate a <em>video capture </em>of an episode played by your trained Qnetwork at different points of the training process (0<em>/</em>3, 1<em>/</em>3, 2<em>/</em>3, and 3<em>/</em>3 through the training process) of both environments. We provide you with a helper function to create the required video captures in test video().</li>

</ul>

<ul>

 <li><strong>(20 pts, optional) </strong>Implement any of the modifications below. Describe what you implemented, and run some experiments to determine if the modifications yield a better RL algorithm. You may implement multiple of the modifications, but you will not receive more than 20 points of extra credit.

  <ul>

   <li>(20 pts) Double DQN, as described in [4].</li>

   <li>(20 pts) Dueling DQN, as described in [5].</li>

   <li>(20 pts) Residual DQN, as described in [1].</li>

  </ul></li>

</ul>

<h1>Guidelines on References</h1>

We recommend you to read all the papers mentioned in the references. There is a significant overlap between different papers, so in reality you should only need certain sections to implement what we ask of you. We provide pointers for relevant sections for this assignment for your convenience.

The work in [2] contains the description of the experimental setup. Algorithm 1 describes the main algorithm. Section 3 (paragraph 3) describes the replay memory. Section 4 explains preprocessing (paragraph 1) and the model architecture (paragraph 3). Section 5 describes experimental details, including reward truncation, the optimization algorithm, the exploration schedule, and other hyperparameters). The methods section in [3], may clarify a few details so it may be worth to read selectively if questions remain after reading [2].

<h1>Guidelines on Hyperparameters</h1>

In this assignment you will implement improvements to the simple update Q-learning formula that make learning more stable and the trained model more performant. We briefly comment on the meaning of each hyperparameter and some reasonable values for them.

<ul>

 <li>Discount factor <em>γ</em>: 1<em>.</em>0 for MountainCar, and 0<em>.</em>99 for CartPole.</li>

 <li>Learning rate <em>α</em>: 0<em>.</em>001 for Cartpole and 0<em>.</em>0001 for Mountaincar.</li>

 <li>Exploration probability-greedy: While training, we suggest you start from a high epsilon value, and anneal this epsilon to a small value (0<em>.</em>05 or 0<em>.</em>1) during training. We have found decaying epsilon linearly from 0<em>.</em>5 to 0<em>.</em>05 over 100000 iterations works well. During test time, you may use a greedy policy, or an epsilon greedy policy with small epsilon (0<em>.</em>05).</li>

 <li>Number of training episodes: For MountainCar-v0, you should see improvements within 2000 (or even 1000) episodes. For CartPole-v0, you should see improvements starting around 2000 episodes.</li>

</ul>

Look at the average reward achieved in the last few episodes to test if performance has plateaued; it is usually a good idea to consider reducing the learning rate or the exploration probability if performance plateaus.

<ul>

 <li>Replay buffer size: 50000; this hyperparameter is used only for experience replay. It determines how many of the last transitions experienced you will keep in the replay buffer before you start rewriting this experience with more recent transitions.</li>

 <li>Batch size: 32; typically, rather doing the update as in (2), we use a small batch of sampled experiences from the replay buffer; this provides better hardware utilization. In addition to the hyperparameters:</li>

 <li>Optimizer: You may want to use Adam as the optimizer. Think of Adam like a fancier SGD with momentum, it will automatically adjust the learning rate based on the statistics of the gradients its observing.</li>

 <li>Loss function: you can use Mean Squared Error.</li>

</ul>

The implementations of the methods in this homework have multiple hyperparameters. These hyperparameters (and others) are part of the experimental setup described in [2, 3]. For the most part, we strongly suggest you to follow the experimental setup described in each of the papers. [2, 3] was published first; your choice of hyperparameters and the experimental setup should follow closely their setup. We recommend you to read all these papers. We have given pointers for the most relevant portions for you to read in a previous section.

<h1>Guidelines on implementation</h1>

This homework requires a significant implementation effort. It is hard to read through the papers once and know immediately what you will need to be implement. We suggest you to think about the different components (e.g., replay buffer, Tensorflow or Keras model definition, model updater, model runner, exploration schedule, learning rate schedule, …) that you will need to implement for each of the different methods that we ask you about, and then read through the papers having these components in mind. By this we mean that you should try to divide and implement small components with well-defined functionalities rather than try to implement everything at once. Much of the code and experimental setup is shared between the different methods so identifying well-defined reusable components will save you trouble.

We provide some code templates that you can use if you wish. Contrary to the previous assignment, abiding to the function signatures defined in <strong>these templates is not mandatory you can write your code from scratch if you wish</strong>.

<h1>References</h1>

<ul>

 <li>Leemon Baird. Residual algorithms: Reinforcement learning with function approximation. In <em>Machine Learning Proceedings 1995</em>, pages 30–37. Elsevier, 1995.</li>

 <li>Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. <em>arXiv preprint arXiv:1312.5602</em>, 2013.</li>

 <li>Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. <em>Nature</em>, 518(7540):529–533, 2015.</li>

 <li>Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. 2016.</li>

 <li>Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. Dueling network architectures for deep reinforcement learning. <em>arXiv preprint arXiv:1511.06581</em>, 2015.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://www.cmu.edu/policies/">https://www.cmu.edu/policies/</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://gym.openai.com/envs/FrozenLake-v0">https://gym.openai.com/envs/FrozenLake-v0</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> After this assignment, we highly recommend that you look at DQN implementations on Github to see how others have structured their code.

<a href="#_ftnref4" name="_ftn4">[4]</a> You can use the Monitor wrapper to generate both the performance curves and the video captures.