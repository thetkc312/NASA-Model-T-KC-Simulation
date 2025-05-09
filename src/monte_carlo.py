"""
Monte Carlo simulation to determine panel misalignment propagation in a linked panel
array, and potential interference between series chains of panels in a full flasher.
I will start by discussing naming conventions for panels and their links, then describe
the set of panels I will use in the simulation.

NAMING CONVENTIONS:
A flasher involves 25 panels linked in an array around a central pentagonal panel, Identified here simply as P.

A typical naming convention for panels is to group them into petals, whose member panels are named as:
A1, A2, A3, B1, B2

I have often identified the panels in a petal that appears immediately counterclockwise
to this petal with a comma after each panel name. Thus, they would be labeled as:
A1, A2, A3, B1, B2,

I have often identified the panels in a petal that appears immediately clockwise to
this petal with an apostraphe after each panel name. Thus, they would be labeled as:
A1' A2' A3' B1' B2'

I will be using a new naming convention for the panels in the simulation. The petal arrangement
does not correspond to the flow of kinematic couplings between panels, whereas this convention does.
Moving radially and outward from the center of the flasher, the panels are labeled as:
S1 S2 S3 S4 S5

The panels in the series chain counterclockwise to this chain are labeled as:
S1- S2- S3- S4- S5-
The panels in the series chain clockwise to this chain are labeled as:
S1+ S2+ S3+ S4+ S5+

MY CHOICE OF PANELS:
Three error propagation panel arrays are used in the simulation.
Maximal angular and translational displacements are measured for the first array,
and while they may be measured for the second and third arrays, their primary purpose
is to observe their relative misalignment as adjacent series chains to the first array.

"""