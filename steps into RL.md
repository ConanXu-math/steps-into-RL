# Reinforcement Learning

* [RL]https://www.bilibili.com/video/BV1r3411Q7Rr?vd_source=5002bb146c6f86977323df7568cda7ca

<img src=".\fig\image-20250717144807205.png" style="zoom:80%;" />

# Basic Concepts

<img src=".\fig\image-20250717144937547.png" style="zoom: 67%;" />

- **State: ** The status of the agent with respect to the environment (locations, ...)

- **State space: ** the set of all states $\mathcal S = \{s_i\}$

- **Action: **For each state, actions that can be taken $a_i$

- **Action space of a state: ** the set of all possible actions of a state $\mathcal A(s_i) = \{a_i\}$

- **state transition: ** moving from one state to another, e.g. $s_1 \xrightarrow{a_i}s_2$  

<img src=".\fig\image-20250717145048887.png" style="zoom: 33%;" />

​	(Tabular representation, probability representation)

​	**Math:**
$$
\begin{cases}
p(s_2 \mid s_1, a_2) = 1 \\
p(s_i \mid s_1, a_2) = 0, \quad \forall i \ne 2
\end{cases}
$$
​	The state transition could be **stochastic**

- **Policy:**  tell the agent what actions to take at a state

<img src=".\fig\image-20250717083117947.png" alt="image-20250717083117947" style="zoom:50%;" />

​	Mathematical Representation Using conditional probability.

​	For example, for state $s_1$:

$$
\begin{aligned}
\pi(a_1 \mid s_1) &= 0 \\
\pi(a_2 \mid s_1) &= 1 \\
\pi(a_3 \mid s_1) &= 0 \\
\pi(a_4 \mid s_1) &= 0 \\
\pi(a_5 \mid s_1) &= 0
\end{aligned}
$$

​	It is a **deterministic** policy. In a stochastic situation, use sampling.

> **Policy** is how the agent chooses actions. 
>  	**State transition** is how the environment responds to actions.
>
> The **agent** uses its **policy** to pick actions.
>
> The **environment** uses **state transitions** to return the next state.

- **Reward:** a real number we get after taking an action. 

$$
R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R} \quad r = R(s,a)
$$

 	A **positive** reward represents **encouragement** to take such actions.

​	A **negative** reward represents **punishment** to take such actions.

​	Reward depends on the state and action but not the next state.

​	Mathematical Description: Conditional Probability

​	**Intuition**: At state $s_1$, if we choose action $a_1$, the reward is $-1$.

​	**Math**: $p(r = -1 \mid s_1, a_1) = 1$ and $p(r \ne -1 \mid s_1, a_1) = 0$ ....

-  **trajectory: **  a state-action-reward chain (evaluate whether a policy is good or not ):

$$
s_1 \xrightarrow{a_2 \atop r=0} s_2 \xrightarrow{a_3 \atop r=0} s_5 \xrightarrow{a_3 \atop r=0} s_8 \xrightarrow{a_2 \atop r=1} s_9
$$

-  **return:**  the sum of all the rewards collected along the trajectory:

$$
\text{return} = 0 + 0 + 0 + 1 = 1
$$

​	A trajectory may be infinite:

$$
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_3} s_8 \xrightarrow{a_2} s_9 \xrightarrow{a_5} s_9 \xrightarrow{a_5} s_9 \cdots
$$

​	The return is:

$$
\text{return} = 0 + 0 + 0 + 1 + 1 + 1 + \cdots = \infty
$$

​	The definition is invalid since the return diverges!

- **discount rate** $\gamma \in [0, 1)$, **Discounted return**:

$$
\begin{aligned}
\text{discounted return} &= 0 + \gamma 0 + \gamma^2 0 + \gamma^3 1 + \gamma^4 1 + \gamma^5 1 + \cdots \\
&= \gamma^3 (1 + \gamma + \gamma^2 + \cdots) = \gamma^3 \cdot \frac{1}{1 - \gamma}
\end{aligned}
$$

​	**Roles of discount factor $\gamma$:**

1. The sum becomes finite.  
2. It balances far and near future rewards.

​	If $\gamma$ is close to 0, the value of the discounted return is dominated by the rewards obtained in the **near future**.

​	If $\gamma$ is close to 1, the value of the discounted return is dominated by the rewards obtained in the **far future**.

- **episode: ** usually assumed to be a **finite** trajectory.  Tasks with episodes are called **episodic tasks**.

## Key elements of Markov Decision Process (MDP):

- **Sets**:
  - **State**: the set of states $\mathcal{S}$
  - **Action**: the set of actions $\mathcal{A}(s)$ is associated with state $s \in \mathcal{S}$
  - **Reward**: the set of rewards $\mathcal{R}(s, a)$

- **Probability distribution**:
  - **State transition probability**:  
    At state $s$, taking action $a$, the probability to transition to state $s'$ is  
    $$
    p(s' \mid s, a)
    $$
  - **Reward probability**:  
    At state $s$, taking action $a$, the probability to get reward $r$ is  
    $$
    p(r \mid s, a)
    $$

- **Policy**:  
  At state $s$, the probability to choose action $a$ is  
  $$
  \pi(a \mid s)
  $$

### **Markov property**: memoryless property

The next state and reward depend **only on the current state and action**, not the full history:

$$
p(s_{t+1} \mid a_{t+1}, s_t, \dots, a_1, s_0) = p(s_{t+1} \mid a_{t+1}, s_t)
$$

$$
p(r_{t+1} \mid a_{t+1}, s_t, \dots, a_1, s_0) = p(r_{t+1} \mid a_{t+1}, s_t)
$$

# Bellman Equation

![image-20250717075911375](.\fig\image-20250717075911375.png)



# Bellman Optimality Equation

![image-20250717072258639](.\fig\image-20250717072258639.png)

# Value Iteration & Policy Iteration

![image-20250717072820088](.\fig\image-20250717072820088.png)

# Monta Carlo Learning

![image-20250717072959774](.\fig\image-20250717072959774.png ) 

# Stochastic Approximation

![image-20250717073407104](.\fig\image-20250717073407104.png)

# Temporal-Difference Learning

![image-20250717073429003](.\fig\image-20250717073429003.png)

# Value Function Approximation

![image-20250717074220107](.\fig\image-20250717074220107.png)

# Policy Gradient Method 

![image-20250717074821949](.\fig\image-20250717074821949.png)

# Actor-Critic Method

![image-20250717075012982](.\fig\image-20250717075012982.png)