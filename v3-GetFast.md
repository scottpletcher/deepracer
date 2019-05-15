# Iteration 3 - GetFast

"GetFast" is a variation of "RacingLine" where I tried to overcome the speed issue.   I had experimented with adding speed factors but wanted to amplify the impact so I squared the speed and used that to multiply the reward.  I hoped the exponential aspect would help develop a policy that really wanted to go fast.

I left all the hyperparameters default and trained for about 8 hours all together over four sepreate 4 hours sessions.  I kept training until it seemed like there was no more increase in the reward trend.  

## Results
Speed improved but not as much as I expected...not enough to be competitive in the Pre-Season virtual league.  I ran this model as a secondary at the AWS Summit in Atlanta and results were poor.  I think my strategy of training until there was no more upward trend in reward meant that the model was overtrained specifically on the simlulator.  When put in a real car, it was ill-equiped to deal.

## Reward Function

```python
def reward_function(params):

    center_variance = params["distance_from_center"] / params["track_width"]

    left_lane = [13,14,15,16,17,18,19,20,21,22,23,24,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
    right_lane = [30,31,32,33,34,35,36,37]
    center_lane = [0,1,2,3,4,5,6,7,8,9,10,11,12,25,26,27,28,29,38,39,68,69]

    reward = 0
    
    if params["all_wheels_on_track"]:
        if params["closest_waypoints"][1] in left_lane and params["is_left_of_center"]:
            reward = 10
        elif params["closest_waypoints"][1] in right_lane and not params["is_left_of_center"]:
            reward = 10
        elif params["closest_waypoints"][1] in center_lane and center_variance < 0.3:
            reward = 10
        else:
            reward = 0.01
    else:
        reward = 0.01
        
    reward *= params["speed"]**2

    return float(reward)
```