# Iteration 5 - Minimalist

What if we took the SelfMotivator concept to the extreme.  [@Carsair](https://github.com/Carsair) posed the [question on this repo](https://github.com/scottpletcher/deepracer/issues/1#issuecomment-512659268) as to whether using progress as the only factor might yeild some interesting results.

I tried this and was really surprised by the results.  A similar function is included in the AWS documentation as a sort of basic example of a reward function.

## Results
I have not tried this model in a physical car, just in the simulator...all on the reInvent 2018 track.  The first hour of training yeilded around 34% completion rate in validation.  In the second hour, I got a 100% and two laps at 65%.  On the third hour, I lowered the default learning rate to 0.0002.  My learning wasn't great but I got 100% for all three validation laps at around 15s.  Just with progress alone!

I think this might be a good "base model" to cross-train on other tracks and add in some speed factors.  I really would like to form a sort of universal model as I think we'll be thrown lots of curves on the tracks at reInvent.

Thus far, I've been treating models as purpose-built tools.  What if I start treating them as things I need to expose to as much variety as possible to make them well-rounded?

## Reward Function

```python
def reward_function(params):

    if params['all_wheels_on_track']:
        reward = params['progress']
    else:
        reward = 0.001

    return float(reward)
```