# Iteration 1 - PurePursuit

This was my first venture outside the examples provided by AWS.  I quite literally opened up a browser and googled ["how to train your self-driving car"](https://www.google.com/search?q=how+to+train+your+self-driving+car&oq=how+to+train+your+self-driving+car).  After sifting through the results, I happened upon an acedemic paper from 1992 by R. Craig Coulter titled ["Implementation of the Pure Pursuit Tracking Algorithm"](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf) and it just made sense to me.

When we drive a real car, we don't look out the side window and ensure we're a distance from the side of the road...rather, we identify a point down the road and use that to orient ourselves.  Pure Pursuit made sense to me so I tried to implement it.

It was developed using SageMaker, RoboMaker and the Jupyter Notebook provided by AWS for those hell-bent on playing with DeepRacer before the console was released.  In that version, the reward function was a little different in how the parameters were passed in so if you want to use this code in the current DeepRacer console, you'll need to modify it a bit.

I left all the hyperparameters default and trained for about 4 hours.  The AWS Summit in Santa Clara was the first time the model ran in a real DeepRacer car and managed to earn [4th place](https://aws.amazon.com/deepracer/schedule-and-standings/leaderboard-santa-clara-summit/) at the end of the day.

I walk through the function in the [presentation I gave at the AWS Summit in Atlanta](https://github.com/scottpletcher/deepracer/blob/master/AWS%20Summit%20ATL%20-%20Deepracer.pdf).

## Reward Function

```python
    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        
        reward = 1e-3
        
        rabbit = [0,0]
        pointing = [0,0]
            
        # Reward when yaw (car_orientation) is pointed to the next waypoint IN FRONT.
        
        # Find nearest waypoint coordinates
        rabbit = [waypoints[closest_waypoints+1][0],waypoints[closest_waypoints+1][1]]
        
        radius = math.hypot(x - rabbit[0], y - rabbit[1])
        
        pointing[0] = x + (radius * math.cos(car_orientation))
        pointing[1] = y + (radius * math.sin(car_orientation))
        
        vector_delta = math.hypot(pointing[0] - rabbit[0], pointing[1] - rabbit[1])
        
        # Max distance for pointing away will be the radius * 2
        # Min distance means we are pointing directly at the next waypoint
        # We can setup a reward that is a ratio to this max.
        
        if vector_delta == 0:
            reward += 1
        else:
            reward += ( 1 - ( vector_delta / (radius * 2)))

        return reward
```