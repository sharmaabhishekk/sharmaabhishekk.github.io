---
name: Expected Threat Derivation in Julia
tools: [Julia, ]
image: "../images/xt_derivation_julia/cover.gif" 
description: Deriving Expected Threat Model using Julia
---

# Implementing Expected Threat (xT) in Julia

Expected Threat (or xT) is one of the coolest ideas to come up in the football analytics sphere in the last few years. Originally introduced by [Karun Singh](https://twitter.com/karun1710) in this [post](https://karun.in/blog/expected-threat.html), it is essentially a ball progression model. That is, it provides us with a framework to value any ball moving action in terms of how likely it is to result in a goal in the next *n* actions (where a good *n* value is typically 4-5). 

Karun has made his xT data available [here](https://karun.in/blog/data/open_xt_12x8_v1.json) and there already exists a python implementation of xT [here](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/xthreat.py). So what's special about this then? My main motivation to put out this post (besides getting more practice with Julia) was to hopefully work through the entire process of converting a raw event data stream to a finished model and, provide a running commentary of sorts on the code.  


## Prerequisites

Assuming you already have Julia installed (I'm currently on version 1.5.2), you'll need the following external libraries

* Plots
* DataFrames
* CSV
* StatsBase

Installing libraries in Julia is fairly simple using the `Pkg` library. 


## Dataset

The dataset we're going to use is the entire last season of Premier League (2019/20). The data is already converted to the SPADL format. If you're not sure what that is, it is basically an attempt to standardize all the different data providers like Wyscout, Statsbomb, Opta, Instat to a single format. I'm personally a big fan as it makes it much easier to work with event data since it is minimal (for more details, go [here](https://github.com/TomDecroos/atomic-spadl#1-conversion-from-event-stream-format-to-spadl)). The csv file is [here](https://github.com/sharmaabhishekk/random_stuff/blob/master/xt_derivation_julia/xt_pre_data.csv). If you're following along with the code, download it and place it in the same folder as your notebook. 

## Getting Started


### The Formula

The final equation as well as the intuition behind it is very well described by Karun in his post. To quote the relevant parts:

> Let $V_{x,y}$ be the "value" that our algorithm assigns to zone $(x,y)$. Now imagine you have the ball at your feet in zone $(x,y)$. You have two choices: shoot, or move the ball. Based on past data, we know that whenever you shoot from here, you will score with probability $g_{x, y}$. Thus, if you shoot, your expected payoff is $g_{x,y}$.

Or, you can opt to move the ball via a pass to a teammate or by dribbling it yourself. But there's another choice to make here: which of the 192 zones should you move it to? Say you choose to move the ball to some new zone, $(z, w)$. In this case, your expected payoff is the value at zone,$(z, w)$, i.e. $V_{z, w}$. But this was just one of the 192 choices that you had; how can we compute the expected payoff for all of the 192 choices in totality? Here's where the move transition matrix $T_{x,y}$ comes in: based on past data, we know where you're likely to move the ball to whenever you're in zone $(x, y)$, so we can proportionally weight the payoffs from each of the 192 zones. Specifically, for each zone $(z, w)$, the payoff is $T_{(x,y)\rightarrow(z,w)} \times V_{z,w}$, i.e. the probability of moving to that zone times the reward from that zone. To get the total expected payoff for moving the ball, we must sum this quantity over all possible zones:

$\sum_{z=1}^{16} \sum_{w=1}^{12} T_{(x,y)\rightarrow(z,w)} \times V_{z,w} $

Finally, let's piece it all together. We computed the payoff if you shoot as $g_{x, y}$, and the payoff if you move the ball as 
 $\sum_{z=1}^{16} \sum_{w=1}^{12} T_{(x,y)\rightarrow(z,w)} \times V_{z,w}$. Based on past data, we know that you tend to shoot $s_{x,y}$ percent of the time, and you opt to move the ball $m_{x,y}$ percent of the time. Therefore, let's weight these two outcomes based on the probability of each of them happening, to obtain our final value for zone $(x, y)$:

$V_{x,y} = (s_{x,y} \times g_{x,y}) + (m_{x,y} \times \sum_{z=1}^{16} \sum_{w=1}^{12} T_{(x,y)\rightarrow(z,w)} V_{z,w})$




## Limitations and Potential Improvements




_______

Hopefully, this post was helpful in some manner! I had a lot of fun working through it. Huge thanks to Michael himself for allowing me to do this plus always patiently helping me with all the questions I had. Also shout out to [Sushruta](https://twitter.com/nandy_sd) and [Maram](https://twitter.com/maramperninety) for reading an earlier version of this draft. :heart:

For any kind of feedback(suggestions/questions) about this post or the code, feel free to reach out to me on Twitter or drop me a mail(the former is always faster).














