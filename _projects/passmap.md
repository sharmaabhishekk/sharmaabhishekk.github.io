---
name: Creating Passmaps in Python
tools: [Python, Matplotlib, Passmaps, Statsbomb]
image: "../images/movie.gif"
description: In this post, we'll try to create a passmap in Python using Matplotib and Pandas.
---


# Creating a Passmap in Python

Passmaps are one of the most popular visualizations in football right now. And for good reason. They pack a lot of useful information
about a single match in an intuitive manner. Passing trends, networks, players' roles in a given system, and even how well they're
performing said roles.

In this blog post, we'll go through the steps to creating your own in Python using Statsbomb's open data.

(If you're just interested in the code, the github link's [here](https://github.com/sharmaabhishekk))

### Pre-requisites

I'm gonna be using Python so you'll need that installed on your system to follow along. If you don't already, you can go over to
[python](python.org) and get it for your system.

Other than that, we'll also we using the following Python libraries:

 Matplotlib - for the actual plotting,
 Pandas - wrangling the data,
 Requests - making a request to get the data,
 and Numpy - some more computing on the data

 All of those should be just a pip install away!

 ### Dataset

 To create a passmap for a match, we'll need some event data. Statsbomb have you covered with their excellent free data. If you don't
 have a local copy of the data, don't worry - that's what the requests library was for.

 ### Basic Overview

 What really is a passmap? Well, there's a lot going on here (and that put me off a bit the first time I saw these in the wild) but
 let's take a closer look at what information it's supposed to convey to us.
 There are two important things which this shows - the average position of the player, and the number of passes between any
 two given players. Apart from that, we also have the players' names, and the players' dot sizes (which indicate the total number of
 passes played by the player). Finally we have some aesthetic details - the watermark, team's logo, match details.
 For the purpose of this post, we are going to ignore the watermark and the logo of the team.

 ### Getting started

 ##Imports

 ```python

import json
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pandas import json_normalize
import numpy as np
from Pitch.pitch import Pitch

 ```

Statsbomb has a unique match_id for every match in the open-data repository. The match we're going to look at is the FIFA WC 2018 Final
between France and Croatia. The id for it is "8658" and let's look at Croatia to start with (which is the away side in the match file).
Let's set some variables to that data and then also get our figure and axis instances from matplotlib.

```python

match_id = "8658"
side = "away"

fig, ax = plt.subplots()
ax = Pitch(ax)

```

The next step would be to write a Class called Player. Why do that? Well, if you think about it a player is basically an object
with certain attributes - name, a unique player_id, average position on the pitch, and the total number of passes attempted completed.

```python

class Player:
    def __init__(self, player, df):
        self.id = player["player"]["id"]
        self.name = player["player"]["name"]
        self.average_position(df)

    def average_position(self, df):

        player_pass_df = df.query("(type_name == 'Pass') & (pass_type_name not in ['Free Kick', 'Corner', 'Throw-in', 'Kick Off']) & (player_id == @self.id) & (pass_outcome_name not in ['Unknown','Out','Pass Offside','Injury Clearance', 'Incomplete'])")
        self.x, self.y = np.mean(player_pass_df['location'].tolist(), axis=0)

        self.n_passes_completed = len(player_pass_df)

```

## Loading the data

We can either load the data from the Github repository online or from your local copy of it. Either way, let's write a function to
take care of both cases. We're going to tell the function which match (match_id), and how to get the data (remote/local). It's gonna return
the data (which is going to be in JSON format) and also the data formatted to a Pandas dataframe.

```python

def load_file(match_id, getter="remote", path = None):

    if getter == "local":
        with open(f"{path}/{match_id}.json", "r", encoding="utf-8") as f:
            match_dict = json.load(f)
            df = json_normalize(match_dict, sep="_")
        return match_dict, df

    elif getter == "remote":
        resp = requests.get(f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json")

        match_dict = json.loads(resp.text)
        df = json_normalize(match_dict, sep="_")

        return match_dict, df

```


















<!-- ![preview](https://www.sketchappsources.com/resources/source-image/we-were-soldiers-landing-page-dbruggisser.jpg)

## Search Movies

![search](https://www.sketchappsources.com/resources/source-image/microsoft-windows-10-virtual-keyboard-diogo-sousa.png) -->

<p class="text-center">
{% include elements/button.html link="https://github.com/YoussefRaafatNasry/portfolYOU" text="Learn More" %}
</p>
