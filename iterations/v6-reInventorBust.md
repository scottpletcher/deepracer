# Iteration 6 - reInvent or Bust

Ok, after much trial and error, I'm going to try an alternative strategy.  I'm going to assume that there will be a gauntlet of tracks at reInvent which will require either a nice variety of models or one pretty universal model.  If I start with a basic model and just train a little over each track, maybe the neural net will develop to handle a variety of scenarios.

I trained across the basic Oval, reInvent track, Bowtie, Empire, Shanghai and Cumulo for 1 hour each, cloning for each iteration.  Adjusted the batch size to 128, switched to Huber loss after Empire.  Left all other hyperparameters the same.

## Results
You'll just have to wait until re:Invent to see.... :)

UPDATE: I managed to miss out on making to the "Round of 16" by just a few tenths of a second.  You can see the whole sordid story here:  [DeepRacer: At the Apex](https://www.youtube.com/watch?v=zVmFobnPP2U&list=PLI1_CQcV71RkakfA9Ar59soxRCkCmkPSt&index=8)

## Reward Function

```python
def reward_function(params):

    reward = 0.001

    if params["all_wheels_on_track"]:
        reward += 1
    if abs(params["steering_angle"]) < 5:
        reward += 1
   
    reward += ( params["speed"] / 8 )
   
    return float(reward)
```
