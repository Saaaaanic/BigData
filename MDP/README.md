# Results

Here is a simple showcase of how to compute optimal policy

This is a game where bot starting at (0, 0) needs to get to the (4, 4) point with obstacles

Using the algo from the file u provided, we can compute the action in each state that needs to be done to get to the end
maximizing our result. There are obstacles that will provide bot with negative reward, so he will avoid it.
 
Plus each step in rewarded with -0.1, so he will find the best optimal path to the end.

_sadly in this situation I can't use algo 6 because policy always will give the same result over time and
will be more like backtracking task rather than learning, and I am ~~lazy:(~~_