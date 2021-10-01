---
title: Futbolista.jl
tags: [Software Development, Julia, Open source]
style: fill
color: success
description: A Julia package to help with some basic football analytics workflows
---
**Github Repository:** [https://github.com/sharmaabhishekk/Futbolista.jl](https://github.com/sharmaabhishekk/Futbolista.jl)

**JuliaHub Package Page:** [https://juliahub.com/ui/Packages/Futbolista/hDjuW/0.2.0](https://juliahub.com/ui/Packages/Futbolista/hDjuW/0.2.0)

Futbolista.jl is a Julia package with some utility functions to speed up some of the most common football analytics related workflows for any analyst. 
I initially wrote it as a collection for utility functions that I needed a lot for basic analytics stuff (loading in data, plotting data, running basic models) but later on decided to clean up the code and put it all on the Julia registry to be a proper package. 

### What can you do with it?

* Load in data (Statsbomb)
* Plot stuff (events, passmaps, pitches)
* Some helper functions for tracking data (pitch control etc)

### Install

```julia
using Pkg
Pkg.add("Futbolista")
```

The package is under heavy development. I hope to add a lot more functionality to it in the upcoming months!