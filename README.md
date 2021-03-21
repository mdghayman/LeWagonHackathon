# LeWagonHackathon
Group 28 entry for Le Wagon Hackathon March 2021, solving the problem from Tray.io

## The problem
Algorithmically predict the SaaS product that will generate more interest and traffic, to proactively create integrations for unicorns.

## The solution
A clustering algorithm which uses 6 attributes to place each prospective API into one of three groups, thereby determining its likelihood of success.

## Additional exploratory analysis
* The `Competitive gap` variable was dropped, as it contained only `False` values and this yielded no insights.
* Of the remaining 6 variables, there was a very close correlation between `SEO value` and `Organic search volume`. There were no other significant correlations among the variables.
* A decision tree was built to determine the conditions in which the `Growing market` variable would be `True`. The results suggested that a high `Persona value` would be most significant in leadning to growing market.
