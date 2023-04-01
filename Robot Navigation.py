import numpy as np
import random
import matplotlib.pyplot as plt


rewardo = []
episodes = []
average_reward = 0
#define the shape of the environment (i.e., its states)
environment_rows = 10
environment_columns = 10

#Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a) 
q_values = np.zeros((environment_rows, environment_columns, 5))

#action codes: 0 = up, 1 = right, 2 = down, 3 = left, 4 = pickup
actions = ['up', 'right', 'down', 'left','pick']

#define a function that will choose a random, non-terminal starting location
def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  
  return current_row_index, current_column_index

#define an epsilon greedy algorithm 
def get_next_action(current_row_index, current_column_index, epsilon):

  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(5)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  reward = 0
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'up' and current_row_index == 0:
    reward -= 5
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'right' and current_column_index == environment_columns - 1:
    reward -= 5
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'down' and current_row_index == environment_rows - 1:
    reward -= 5
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  elif actions[action_index] == 'left' and current_column_index == 0:
    reward -= 5
  
  return new_row_index, new_column_index, reward

def train(epsilon,discount_factor, learning_rate):
  #Train the Agent
  cans = 0
  can_counter = 0
  #5000 episode training
  for episode in range(10000):

    #get the starting location for this episode
    row_index, column_index = get_starting_location()
    reward = 0

    #Create a 2D numpy array to hold the rewards for each state. 
    
    rewards = np.full((environment_rows, environment_columns), 0)
    
    for i in range(len(rewards)):
        for j in range(len(rewards)):
            if random.random() <0.5:
                rewards[i][j] = 10
                cans+=1

    #start training robby 
    for i in range(200):

      #if robot done picking up cans
      if(cans == can_counter):
        break
      
      #choose which action to take
      action_index = get_next_action(row_index, column_index, epsilon)

      old_row_index, old_column_index = row_index, column_index 
      row_index, column_index, increment = get_next_location(row_index, column_index, action_index)

      reward += increment
      if(action_index == 4 and rewards[row_index, column_index] != 10):
        reward -= 1
      
      #receive the reward for moving to the new state, and calculate the temporal difference
      #remove reward from table to simulate pickup
      if(rewards[row_index, column_index] == 10):
         can_counter+= 1
         
      reward += rewards[row_index, column_index]

      rewards[row_index, column_index] = 0

      old_q_value = q_values[old_row_index, old_column_index, action_index]
      temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

      #update the Q-value for the previous state and action pair
      new_q_value = old_q_value + (learning_rate * temporal_difference)
      q_values[old_row_index, old_column_index, action_index] = new_q_value
    
    #every 50 episodes epsilon adjusted to decrease randomness
    if(episode % 50 ==0 and epsilon < 1):
      epsilon+= 0.05
    
    if('''episode% 100 ==0'''):
      rewardo.append(reward)
      episodes.append(episode) 

  print('Training complete!')
  plt.plot(episodes, rewardo)
  plt.show()


  return q_values

def test(epsilon):
  row_index, column_index = get_starting_location()
  reward = 0
  cans = 0
  can_counter = 0
  rewards = np.full((environment_rows, environment_columns), 0)
    
  for i in range(len(rewards)):
      for j in range(len(rewards)):
          if random.random() <0.5:
              rewards[i][j] = 10
              cans+=1
  
  for i in range(200):
      if(cans == can_counter):
        print(i)
        break
      #choose which action to take
      action_index = get_next_action(row_index, column_index, epsilon)

      #old_row_index, old_column_index = row_index, column_index 
      row_index, column_index, increment = get_next_location(row_index, column_index, action_index)

      reward += increment
      if(action_index == 4 and rewards[row_index, column_index] != 10):
        reward -= 1
        
      #receive the reward for moving to the new state
      if(rewards[row_index, column_index] == 10):
         can_counter+= 1

      reward += rewards[row_index, column_index]

      rewards[row_index, column_index] = 0
      


  return reward



#runs the main function to train the agent
#epsilon,discount_factor, learning_rate
trained_Q_matrix = train(0.8,0.9,0.5)
print(trained_Q_matrix)
#runs the test agent which uses the updated q-table
print(test(0.9), "rewarded from test")



