# Motion filtering

The basic algorithm for detecting motion used here is based on the fast vs. slow mechanism for computing a temporal derivative, where more recent changes are reflected in the fast accumulating process, and older changes are in the slow one. This is similar to the tapped delay line mechanism, but the temporal derivative mechanism is the more general principle.

```
  O o-o O
+||-   -||+
 O       O 
```

In the above diagram, the receiving neuron at the top receives balanced fast excitatory + connections and slow inhibitory connections - with the same net synaptic strength. Two receiving neurons from spatially adjacent locations mutually inhibit eachother. When there is no motion, the excitation and inhibition cancel out.

When visual input is moving from the right to the left, the right neuron experiences a net inhibitory signal because the slower inhibition persists longer than the fast excitation, while the opposite is true of the left neuron. The opposite pattern holds for motion in the opposite direction. Thus, the relative balance between these paired neurons provides a readout of the motion direction. 

Critically, when there is a uniform onset of a novel input, both neurons receive the same initial excitatory transient, which then is balanced by the slower inhibition. Their mutual inhibition cancels this out.

The circuit also needs to multiply by the min activity of any input, so that it doesn't just respond to edges where there is nothing and then something.

Key that integration has immediate rise and the difference is in the decay times, so anything that is still is always equalized at the same values.

# Retinal direction filtering

The starburst amacrine cells (SAC) are critical for computing directionally-sensitive signals in the retina (Wei, 2018; Morrie & Feller; 2018; Brombas et al, 2017; Jain et al, 2020; Greene et al, 2016). They have radially-symmetric dendritic arbors, and exhibit center-out motion sensitivity, such that visual signals propagating from the center RF outward maximally activate the neuron. The maximal activating stimulus would thus be an annular looming-like stimulus, e.g., zooming into a ring-shaped stimulus centered around the central RF point.

Similar to the radially-symmetric DoG cells, radially-symmetric detectors like this can also be combined to produce overall directionally specific detectors, by offsetting the preferred and null regions of different SAC inputs.

The distal dendritic branches appear to support something like the above circuit, receiving inputs from bipolar cells with different temporal dynamics, and having independent integration dynamics within each distal branch. Thus, the overall response as these distal branches are aggregated is a spatial integration of the independent motion signals for smaller sub-regions.

It does not appear that there is a single dominant computational model in this area, according to an admittedly limited search.

