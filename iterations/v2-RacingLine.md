# Iteration 2 - RacingLine

This was the first model developed entirely under the DeepRacer console.  One of my colleagues noticed that the reward function now provided us with an indicator as to which lane the car is in (```is_left_of_center```).

Given the waypoints provided by the [AWS Deepracer Workshop repo](https://github.com/aws-samples/aws-deepracer-workshops/blob/master/Workshops/2019-AWSSummits-AWSDeepRacerService/Lab1/img/reinventtrack_waypoints.png), we could trace an optimized racing line around the track.  Then, we could create a function that incentivizes the car to be in the right lane, left lane or near the center of the track, which would theoretically mimic our intended racing line.

I also tried to introduce some negative reinforcement here if the car is off the track, but it wasn't really true negative reward as the function would always return a positive number because I started with ```reward = 21```.

I left all the hyperparameters default and trained for about 8 hours all together over four sepreate 4 hours sessions.  I kept training until it seemed like there was no more increase in the reward trend.

## Results
The model could consistently complete laps around the reInvent track in the simulator on the console, but it just wasn't very fast.  I tried to train a model that used full speed actions but it was too unstable and kept flying off the track.  I ran a slight variation of this model ("GetFast") in Atlanta as a secondary and it was really poor in the real car.

## Reward Function

```python
    def reward_function(params):

    center_variance = params["distance_from_center"] / params["track_width"]

    left_lane = [13,14,15,16,17,18,19,20,21,22,23,24,40,41,42,43,44,45,46,50,51,52,53,60,61,62,63,64,65,66,67,68,69]
    right_lane = [33,34,35,36,37]
    center_lane = [0,1,2,3,4,5,6,7,8,9,10,11,12,25,26,27,28,29,30,31,32,38,39,47,48,49,54,55,56,57,58,59]
    
    reward = 21

    if params["all_wheels_on_track"]:
        reward += 10
    else:
        reward -= 10

    if params["closest_waypoints"][1] in left_lane and params["is_left_of_center"]:
        reward += 10
    elif params["closest_waypoints"][1] in right_lane and not params["is_left_of_center"]:
        reward += 10
    elif params["closest_waypoints"][1] in center_lane and center_variance < 0.4:
        reward += 10
    else:
        reward -= 10
    
    return float(reward)
```