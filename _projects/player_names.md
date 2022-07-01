---
author: abhisheksh_98
name: How Many Gabriels Can You Fit in a Team?
tools: [Python, Pandas,  FbRef, Scraping]
image: "../images/player_names/cover.jpg" 
description: Building a Starting XI of Players With the Same Name
date: 2022-06-28
---

# Building a Starting XI With All Players With the Same Name

I love the transfer window! So many possibilities. Like which players will Manchester City get on their budget of the GDP of a small country. Or which player will make the mandatory annual surprise move to the MLS. Or...how many Gabriels can a club fit in their squad? 

With the signing of Gabriel Jesus, Arsenal now have 4 players named Gabriel in the squad. This got me thinking: 

<span style="color: salmon;font-weight: bold;font-size: 35px;text-align: center;">What’s the best starting XI you can create with all 
players having the same name?</span>

Best is definitely subjective so to further specify the problem statement, let's lay out some basic criteria:
* <span class="highlight-text">Male</span> players from <span class="highlight-text">all eras are eligible</span> - if only to get more names to pick from.
* The name we pick has to be in the names of each of our 11 players - but it can <span class="highlight-text">anywhere in the name (first, last, or middle)</span>. So Daniel James and James McGinn would both be eligible for James XI.
* Latin diacritics in the <span class="highlight-text">names can be simplified to their english ASCII versions</span>. Michaël Cuisance is eligible for Michael XI.
* But, <span class="highlight-text">hypocorisms or shortened versions of the name aren't allowed</span> if that is the exact first name. So Mikel Arteta is not eligible for Michael FC but Dani Parejo is eligible for Daniel FC because Dani is just short for his name Daniel.
* The final XI has to be a reasonably <span class="highlight-text">functional XI</span> - you can't make an XI of just defenders or goalkeepers.

[*Seems like a reasonable set of criteria but spoiler alert, even this will be challenging with the data we have.*] 

------

## Scraping

To answer this question, we’d need data. Mainly player names - lots of them - but also some basic biography about them - leagues, teams, position, years active etc. Since I don't have access to any API, I have no choice but to scrape. 

I scoured the most common public hunting grounds - transfermarkt, whoscored, sofascore. In the end, Wikipedia and FbRef seem like the best options. 

* Wikipedia has a page [here](https://en.wikipedia.org/wiki/Category:Lists_of_association_football_players) which contains lists of lists-of-pages of football players. 

* FbRef has a player index page [here](https://fbref.com/en/players/) which points to pages where players are segregated by their intitals. 

I decided to get the data from fbref as it would be the easiest and fastest to scrape and also has a little more biography data. The index page is close to ideal as a starting point for us. There's some **summarized** data for the players (strong emphasis on summarized because it is very minimal) and that's what we'll capture.

<br>
<details>
<summary class='highlight-text-summary'>
Code cell
</summary>

{% highlight python %}
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pandas as pd

r = requests.get("https://fbref.com/en/players/")
soup = BeautifulSoup(r.content, "lxml")
letter_links = soup.select("ul.page_index a")

players_data = []
for link in tqdm(letter_links, desc="Getting letter pages..."):
    time.sleep(2.5) ##make sure we are being polite and following the rate limit
    r = requests.get("https://fbref.com"+link['href'])
    soup = BeautifulSoup(r.content, 'lxml')

    ptags = soup.select("div[id^='div_'].section_content p")
    for ptag in tqdm(ptags, desc="Collecting player information..."):
        try:
            player_dict = {}
            player_dict['link'] = ptag.select_one("a")['href']
            player_dict['player_name'] = ptag.select_one("a").text
            player_dict["other_details"] = ptag.getText().split("·")[1:]
            players_data.append(player_dict)
        except error as e:
            print(e)

df = pd.DataFrame(players_data)
df = pd.to_csv("players_data.csv" index=False)
{% endhighlight %}

</details>
<br>

What we are doing here is pretty simple. We'll use the page links from the index page, visit each link, and collect the player names and other details from there. We finally save it to a csv file. With the 2.5 second sleep we have, the entire script should take approximately 25 minutes.

The data for each player is summarized so what we end up collecting has some major limitations. Namely, 

a) It <span class="highlight-text">doesn’t have full names</span>. This is important because it gives us more options. For example, Luke Ayling isn't eligible for David XI but Luke David Ayling definitely is.

b) It <span class="highlight-text">doesn’t have any league information</span>. It does have team names though so we can attempt to work that part out. 

c) There's <span class="highlight-text">no information about the player's career</span> beyond how long they were active for. How successful they've been, how popular they are and so on. 

d) The <span class="highlight-text">position tags are only basic</span>. For example, we have only **DEF** for all defenders, including center-backs and fullbacks. This makes things <s>impossible</s> harder to aumotate.

[*Unfortunately, this is probably the best we can do without either hitting each player page (which would take a ridiculous amount of time) or getting access to an API. It is still a decent starting point, and considering this is just a random 2 AM idea, I am happy enough.*]

Time for some data cleaning and EDA.

-------

## Data Cleaning

<br>
<details>
<summary class='highlight-text-summary'>
Code cell
</summary>

{% highlight python %}
import pandas as pd; pd.set_option('display.max_columns', None)
import numpy as np

df = pd.read_csv("players_data.csv")
df.head()

"""
|    | link                                 | player_name     | other_details                                                     |
|---:|:-------------------------------------|:----------------|:------------------------------------------------------------------|
|  0 | /en/players/9c60f681/Ahmad-Aadi      | Ahmad A'adi     | ['2014-2022', 'GK', 'Kuwait']                                     |
|  1 | /en/players/ad713dff/Jamal-Aabbou    | Jamal Aabbou    | ['2018-2022', 'FW', 'R.E. Virton, Lommel SK, Lierse Kempenzonen'] |
|  2 | /en/players/c2e5d028/Zakariya-Aabbou | Zakariya Aabbou | ['2021-2022', 'FW', 'RC Vichy']                                   |
|  3 | /en/players/c48b5529/Kim-Aabech      | Kim Aabech      | ['2007-2017', 'MF', 'Lyngby, Aarhus, Horsens, Nordsjælland']      |
|  4 | /en/players/d7ed844d/Kamilla-Aabel   | Kamilla Aabel   | ['2018-2021', 'DF', 'Røa, Arna-Bjørnar']                          |
"""

df.shape 

"""
(169623, 3)
"""
{% endhighlight %}
</details>

<br>

We've got data on almost 1,70,000 players. 

The `other_details` column contains data about the **years the player played**, the **position**, and their **team(s)** - club or national team. It is a column of lists but they are all represented as strings. Also, not all rows have a list of those exact 3 strings so we need to clean it up. 

```python
df["other_details"].apply(eval).apply(len).value_counts()

3    122609
2     42524
1      2665
0      1819
4         6
```

We'll have to deal with all of these in a case-by-case basis. 

Some of the other data cleaning steps to finish:

* Replace the latin diacritics with their ASCII equivalents
* Split the names out to check which names are the most common amongst players
* Split years to get starting and ending years

[*Data cleaning isn't the most fun thing in the world to do, so you can skip the next code cell to go straight to the results.*]

<br>
<details>
<summary class='highlight-text-summary'>
Code cell
</summary>

{% highlight python %}

def clean_data(data):
    """
    """
    if data.strip()[0] == "[" and data.strip()[-1] == "]":
        data = eval(data) 
        data = [value.strip() for value in data]
        
        if len(data) == 4:
            a,b,c,d = data
            return [a,b,c+d]

        elif len(data) == 3: ## all three values in list
            return data

        elif len(data) == 2: ##when two values in list
            a,b = data
            if a.startswith("20") or a.startswith("19"):
                if b in ["MF", "DF", "GK", "FW"]:
                    return [a, b, np.nan]
                else:
                    return [a, np.nan, b]
            elif a in ["MF", "DF", "GK", "FW"]:
                return [np.nan, a, b]
            else:
                return [np.nan, np.nan, np.nan]

        elif len(data) == 1: ##case where only 1 value in list
            a, = data
            if a.startswith("20") or a.startswith("19"):
                return [a, np.nan, np.nan]
            elif a in ["MF", "DF", "GK", "FW"]:
                return [np.nan, a, np.nan] 
            else:
                return [np.nan, np.nan, a]
            
        elif len(data) == 0:
            return [np.nan, np.nan, np.nan]

    else:
        return [np.nan, np.nan, np.nan]
        
        
def get_first_name(x):
    try:
        return x.split(" ", 1)[0]
    except:
        return np.nan
    
def get_last_name(x):
    try:
        return x.split(" ", 1)[1]
    except:
        return np.nan   
    
def get_years(years):
    if years is not np.nan:
        if "-" in years:
            return [int(years.split("-")[0]), int(years.split("-")[1])]
        else:
            return [np.nan, np.nan]
    else:
        return [np.nan, np.nan]

df['other_details'] = df["other_details"].apply(clean_data)
df[['years', 'position', 'squad_nation']] = pd.DataFrame(df['other_details'].tolist(), index=df.index)
df[['start_year', 'end_year']] = pd.DataFrame(df['years'].apply(get_years).tolist(), index=df.index)
df['playing_time'] = df['end_year'] - df['start_year'] 
df['first_name'] = df.player_name.apply(get_first_name)
df['last_name'] = df.player_name.apply(get_last_name)
df.head()

"""
|    | link                                 | player_name     | other_details                                                     | years     | position   | squad_nation                               |   start_year |   end_year |   playing_time | first_name   | last_name   |
|---:|:-------------------------------------|:----------------|:------------------------------------------------------------------|:----------|:-----------|:-------------------------------------------|-------------:|-----------:|---------------:|:-------------|:------------|
|  0 | /en/players/9c60f681/Ahmad-Aadi      | Ahmad A'adi     | ['2014-2022', 'GK', 'Kuwait']                                     | 2014-2022 | GK         | Kuwait                                     |         2014 |       2022 |              8 | Ahmad        | A'adi       |
|  1 | /en/players/ad713dff/Jamal-Aabbou    | Jamal Aabbou    | ['2018-2022', 'FW', 'R.E. Virton, Lommel SK, Lierse Kempenzonen'] | 2018-2022 | FW         | R.E. Virton, Lommel SK, Lierse Kempenzonen |         2018 |       2022 |              4 | Jamal        | Aabbou      |
|  2 | /en/players/c2e5d028/Zakariya-Aabbou | Zakariya Aabbou | ['2021-2022', 'FW', 'RC Vichy']                                   | 2021-2022 | FW         | RC Vichy                                   |         2021 |       2022 |              1 | Zakariya     | Aabbou      |
|  3 | /en/players/c48b5529/Kim-Aabech      | Kim Aabech      | ['2007-2017', 'MF', 'Lyngby, Aarhus, Horsens, Nordsjælland']      | 2007-2017 | MF         | Lyngby, Aarhus, Horsens, Nordsjælland      |         2007 |       2017 |             10 | Kim          | Aabech      |
|  4 | /en/players/d7ed844d/Kamilla-Aabel   | Kamilla Aabel   | ['2018-2021', 'DF', 'Røa, Arna-Bjørnar']                          | 2018-2021 | DF         | Røa, Arna-Bjørnar                          |         2018 |       2021 |              3 | Kamilla      | Aabel       |
"""
{% endhighlight %}
</details>
<br>

Let's find the most common names amongst all players. 

<br>
<details>
<summary class='highlight-text-summary'>
Code cell
</summary>

{% highlight python %}

merged = df.first_name.value_counts().\
            reset_index().\
            rename(columns={"index":"name"}).\
            merge(df.last_name.\
                  value_counts().\
                  reset_index().\
                  rename(columns={"index":"name"}), 
                  on='name', how='outer').\
            fillna(0)
merged['total'] = merged['first_name'] + merged['last_name']
merged = merged.sort_values("total", ascending=False).reset_index(drop=True)
merged.head(35)

{% endhighlight %}
</details>
<br>

|    | name      |   first_name |   last_name |   total |
|---:|:----------|-------------:|------------:|--------:|
|  0 | David     |         1234 |          40 |    1274 |
|  1 | Daniel    |         1121 |          25 |    1146 |
|  2 | Jose      |         1113 |           6 |    1119 |
|  3 | Juan      |          942 |           0 |     942 |
|  4 | Luis      |          813 |          33 |     846 |
|  5 | Kevin     |          815 |           0 |     815 |
|  6 | Carlos    |          763 |          49 |     812 |
|  7 | Martin    |          638 |         154 |     792 |
|  8 | Thomas    |          632 |         143 |     775 |
|  9 | Michael   |          734 |          13 |     747 |
| 10 | Ivan      |          652 |          10 |     662 |
| 11 | Diego     |          632 |           2 |     634 |
| 12 | Marco     |          603 |           2 |     605 |
| 13 | Paul      |          545 |          29 |     574 |
| 14 | Lucas     |          509 |          47 |     556 |
| 15 | Alex      |          542 |           4 |     546 |
| 16 | Christian |          539 |           6 |     545 |
| 17 | Nicolas   |          522 |           9 |     531 |
| 18 | Mohamed   |          464 |          62 |     526 |
| 19 | Antonio   |          461 |          35 |     496 |
| 20 | Jorge     |          448 |          18 |     466 |
| 21 | Sebastian |          463 |           2 |     465 |
| 22 | Jonathan  |          452 |           1 |     453 |
| 23 | Ali       |          374 |          76 |     450 |
| 24 | Victor    |          428 |          19 |     447 |
| 25 | John      |          406 |          30 |     436 |
| 26 | Mario     |          419 |           4 |     423 |
| 27 | Lee       |          365 |          55 |     420 |
| 28 | Luca      |          416 |           2 |     418 |
| 29 | Peter     |          400 |          16 |     416 |
| 30 | Gonzalez  |            2 |         412 |     414 |
| 31 | Gabriel   |          365 |          42 |     407 |
| 32 | Cristian  |          398 |           2 |     400 |
| 33 | Pedro     |          369 |          26 |     395 |
| 34 | Andrea    |          392 |           3 |     395 |

**David** and **Daniel** seem to be our overall best bets with their being more than 100 more Davids than Daniels. The most common last name seems to be **Gonzalez** (check out the `last_name` column) followed by **Lopez** and **Perez**.

Before we go further, let's check where **Gabriel** ranks amongst names. 

```python
merged.query("name == 'Gabriel'").index + 1

##48
```
----------
Intuitively, it makes sense to pick the most common names to make our XI. We have some viable names now. The next step is to make the XI.

Note: If you're a walking player encyclopedia yourself and know more than enough Daniels or Pedros to make a starting XI, feel free to skip the rest of the post. I'm not so I'll try to find viable names from the data I have itself. 

```python
df.start_year.describe()

"""
count    139667.000000
mean       2011.917540
std           8.736951
min        1930.000000
25%        2007.000000
50%        2015.000000
75%        2018.000000
max        2022.000000
"""
```

Seems like the oldest players we have is from 1930. There's a lot of missing data in there and overall the data is very right skewed - which does make sense as the more recent players are natuarlly going to be covered better. 

```
df.position.value_counts(normalize=True)*100

MF       32.931221
DF       27.958792
FW       19.976878
GK       11.107614
FW,MF     3.829648
DF,MF     3.681958
DF,FW     0.446096
DF,GK     0.033896
GK,MF     0.018159
FW,GK     0.015737
```
Almost a third of all players we have are midfielders, followed by defenders and forwards. Note there are some of these mixed position strings (DF,MF/FW,GK) - important to know when we run our queries. 

Now since we have players from all eras and all leagues across the world, I want some way to rank our players. We need this because once I pick a name - say "Clarence", I want to get all the Clarences in our dataset to choose from for our final Clarence XI. <span class="highlight-text">You could go through all Clarences and get 11 names but I want to see what more can I do to have the best names rise up to the top</span>. 

We don't have any player value data (player worth, wages, etc) or which leagues they've played in. To confound matters further, we don't have data on how long they've played for each team either - just the names of the teams they've played for. Nonetheless, I'll create a ghetto rating system based on the limited data we have. It doesn't have to be perfect, it only has to better than the alphabetical order and throw the most obvious names at the top. Here's what I ended up doing:

Decide on three tiers of teams:

* **Tier 1**: The top teams from the top five leagues
* **Tier 2**: The remaining teams from the top five leagues
* **Tier 3**: Rest of the teams

```python
tier_1 = ["Real Madrid",
            "Manchester United",
            "Barcelona",
            "Juventus",
            "Chelsea",
            "Liverpool",
            "Bayern Munich",
            "Arsenal",
            "Paris Saint-Germain",
            "Manchester City",
            "Borussia Dortmund",
            "AC Milan",
            "Napoli",
            "Ajax",
            "Tottenham", 
            "Atlético Madrid",
         ]

tier_2 = ['Marseille', 'Leverkusen', 'Sevilla', 'Monaco','Rennes', 'Nice', 'RB Leipzig', 'Betis', 'Lazio', 'Union Berlin', 
          'Köln', 'Manchester Utd', 'West Ham', 'Nantes', 'Athletic Club', 'Lille', 'Hellas Verona', 'Leicester City', 
          'Mainz 05', 'Hoffenheim', 'Brighton', 'Wolves', 'Torino', 'Sassuolo', "M'Gladbach", 'Newcastle Utd', 
          'Crystal Palace', 'Valencia', 'Brest', 'Udinese', 'Eint Frankfurt', 'Wolfsburg', 'Bochum', 'Osasuna', 
          'Celta Vigo', 'Reims', 'Brentford', 'Bologna', 'Aston Villa', 'Montpellier', 'Augsburg', 'Rayo Vallecano', 
          'Elche', 'Espanyol', 'Angers', 'Empoli', 'Southampton', 'Getafe', 'Cádiz', 'Everton', 'Mallorca', 'Troyes', 
          'Granada', 'Leeds United', 'Stuttgart', 'Hertha BSC', 'Sampdoria', 'Lorient', 'Spezia', 'Clermont Foot', 
          'Burnley', 'Levante', 'Saint-Étienne', 'Arminia', 'Alavés', 'Metz', 'Bordeaux', 'Salernitana', 'Cagliari', 
          'Genoa', 'Venezia', 'Watford', 'Norwich City',
          ]
```
Now, I'll just give each player a <span class="highlight-text">True/False</span> flag for if they've played for any of the teams from `tier_1` or `tier_2`.

```python
df['is_tier_1'] = False
df.loc[(df['squad_nation'].str.contains('|'.join(tier_1))) & (df['squad_nation'] == df['squad_nation']), "is_tier_1"] = True

df['is_tier_2'] = False
df.loc[(df['squad_nation'].str.contains('|'.join(tier_2))) & (df['squad_nation'] == df['squad_nation']), "is_tier_2"] = True
```
--------
## Creating the Starting XI

Finally, we decide on 11 players.

```python
player_name = "David" ##let's try the most common name
player_df = df.query(f"first_name == '{player_name}' or last_name == '{player_name}'")
player_df.sort_values(["is_tier_1", 
                          "is_tier_2",
                          "end_year",
                          "playing_time"], ascending=[False, False, False, False],
                      inplace=True
                     )

for position in ["GK", "DF", "MF", "FW"]:
    filtered = player_df.loc[player_df['position'].str.contains(position).fillna(False)].player_name.tolist() ## this is a better way because of the mixed position strings
    print(position, ": ", filtered[:15], "\n")

"""
GK :  ['David Ospina', 'David de Gea', 'David Cobeno', 'David Seaman', 'David Raya', 'David Stockdale', 'David Marshall', 'David Martin', 'David Button', 'David Soria', 'David Gil', 'David Mitrovic', 'David Oberhauser', 'David Lozancic', 'David Yelldell'] 

DF :  ['David Alaba', 'David Lomban', 'David Lopez', 'David Costas', 'David Rozehnal', 'David Giubilato', 'David Sommeil', 'David May', 'David Burrows', "David O'Leary", 'David Luiz', 'David Mateos', 'David Lee', 'David Brightwell', 'David Kiki'] 

MF :  ['David Alaba', 'David Silva', 'David Lopez', 'David Villa', 'David Pizarro', 'David Odonkor', 'David Hellebuyck', 'David Ginola', 'David Hopkin', 'David Platt', 'David Rocastle', 'David White', 'David Babunski', 'David Luiz', 'David Concha'] 

FW :  ['David Silva', 'David Barral', "David N'Gog", 'David Villa', 'David Bellion', 'David Odonkor', 'David Trezeguet', 'David Ginola', 'David White', 'David Concha', 'David Neres', 'David Drocco', 'David Sesa', 'David Raya', 'David Henen']
"""
```
The goalkeeper position is definitely stacked but lots of solid names in the other positions too. Here's the final 11 names I picked 

<details>
<summary class='highlight-text-summary'>
Code cell
</summary>

{% highlight python %}

from mplsoccer.pitch import VerticalPitch
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Calibri']

p = VerticalPitch('opta',pitch_color='grass', line_color='white', stripe=True)

fig, ax = p.draw(figsize=(8,12))
fig.set_facecolor('green')
ax.set_title(label=f"{player_name} XI",
            fontsize=35, x=0.04,
             y=0.98,
            ha='left',
            fontweight='bold')

xs = [15, 
     40, 30, 30, 40, 
     60, 45, 60, 
     70, 80, 70]
ys = [50, 
     90, 65, 35, 10, 
     65, 50, 35, 
     85, 50, 15]
names = [
    "David de Gea",
"David Raum", "David Alaba", "David Luiz", "Luke David Ayling",
"David Silva", "David Lopez", "David Beckham",
"David Neres", "Jonathan David", "David Brooks",]

ax.scatter(ys,xs, s=1000, color='xkcd:salmon', ec='k',lw=2, marker = MarkerStyle("o", fillstyle="top"))
ax.scatter(ys,xs, s=1000, color='dodgerblue', ec='k',lw=2, marker = MarkerStyle("o", fillstyle="bottom"))

[ax.text(y,x-4, name, va='center', ha='center', fontsize=18, fontweight='demibold') for x,y,name in zip(xs, ys, names)];

{% endhighlight %}
</details>
<br>

<img style="width: auto;height: 800px;display: flex;" src="../images/player_names/david.png" alt="David XI">

> David XI seems pretty solid! 

Perhaps an even more interesting problem would be to combine two less popular names (with less than say, 100 occurences each) to build the best possible starting XI. Some positions might be better stacked in certain names (by random chance if not by nominative determinism) so it would be an interesting optimization problem to get the best possible XI with players from the least popular name pools. You'd need a way to evaluate the final XIs (perhaps something to do with player value) too so perhaps that's a problem for a later blog post?

----------





