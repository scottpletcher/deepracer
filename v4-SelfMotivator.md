# Iteration 4 - SelfMotivator

While digging more into reinforcement learning theory, I happened across a statement that I hadn't considered before and it just blew my mind.  Paraphrasing here...

> With supervised learning, your model will only be as good as the ground truth you have to give it.  With reinforcement learning, the model has the potential to become better than anything or anyone has ever done that thing.

So, rather than be so bold as to think I know the best and fastest way to get that car around the track, what if I just trust in the reinforcement learning process to figure out the best way!  I decided to create the simplest function I possibly could that simply motivated the model to stay on the track and get around in as few steps as possible.   My thinking was that more progress in fewer steps, so long as we're on the track, means a more effiecnt lap.

I trained for about 3 hours and much to my surprise, the car was seemingly finding its own racing lines around the corners!  

## Results
I ran this model as a primary at the AWS Summit in Atlanta and [managed to land 8th place at the end of the day](https://d3akhm1epsal2g.cloudfront.net/bigscreen/?event=atlanta).  In the real car, it was much more smooth than PurePursuit from a steering standpoint.  Not sure if that was reward function at work or just improved DeepRacer code.

## Reward Function

```python
def reward_function(params):

    if params["all_wheels_on_track"] and params["steps"] > 0:
        reward = ((params["progress"] / params["steps"]) * 100) + (params["speed"]**2)
    else:
        reward = 0.01
        
    return float(reward)
```