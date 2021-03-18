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

The dataset we're going to use is the entire last season of Premier League (2019/20). The data is already converted to the SPADL format. I'm personally a big fan of standardized formats like SPADL which make it easy to work with different data providers. The csv file is here; in case you want to follow along, download it in the same folder as your notebook. 

## Getting Started

### The Formula

$h_\theta(x) = \Large\frac{1}{1 + \mathcal{e}^{(-\theta^\top x)}}$


## Limitations and Potential Improvements


Most of the model limitations are mentioned in the, you guessed it, original post. For example, we have no data on defensive activity(though we do know defensive activity is [influenced heavily by opportunity](https://statsbomb.com/2014/06/introducing-possession-adjusted-player-stats/)). Nonetheless, adding possession-adjusted versions of tackles, interceptions, fouls and aerial duels will certainly help us distinguish players who are stylistically different. 

On the topic of data and features, another issue is the absence of spatial data. That would tell us more about *where* on the field do players play. We could also try converting the raw number metrics into team ratios - for example, instead of looking at just number of shots by a player, look at ratio of shots taken by the player to the entire team's. 


The **second** most important limitation is the question about "*high performance within a role vs. misclassification of the player*". For example, a fullback inverting into midfield (looking at you, Kyle Walker) and in general doing more midfielder-y stuff (being more involved in build-up, crossing less) is going to be classified as a midfielder. The best way to go about fixing this would perhaps be a more probabilistic approach to clustering, i.e., predicting that the probability of Walker being a midfielder is 0.45, full-back is 0.5 etc.


The **third** biggest limitation of this method is that there's no regulation on the sizes of the clusters. This combined with the absence of a great evaluation method can (potentially) really mess up your clusters - especially if the features are not carefully chosen too (I can't tell you the number of times I got a cluster full of only Manchester City players after I put in the wrong input). As it is, if you call `np.bincount` on the `labels`, you'll notice that my largest cluster is twice as big as the smallest. Fixing this might also indirectly chip away at the misclassification problem; setting a lower and upper cap on cluster sizes might lead to tighter clusters.

My final idea for possible improvements is to simply try a bunch of other dimension reduction algorithms(UMAP, variational auto-encoders, or even good old PCA), or tweaking the complexity hyper-parameter, and/or increasing the number of components for t-SNE reduction). The last part I did try but the improvement wasn't all that much but with the added downside of it being no longer easy enough to create those scatter plots so I decided to leave it out. 

_______

Hopefully, this post was helpful in some manner! I had a lot of fun working through it. Huge thanks to Michael himself for allowing me to do this plus always patiently helping me with all the questions I had. Also shout out to [Sushruta](https://twitter.com/nandy_sd) and [Maram](https://twitter.com/maramperninety) for reading an earlier version of this draft. :heart:

For any kind of feedback(suggestions/questions) about this post or the code, feel free to reach out to me on Twitter or drop me a mail(the former is always faster).














