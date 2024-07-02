# quadwild-wrapper
An experimental addon wrapping the QuadWild remeshing library.

This came about as a result of TeamC's brief foray into the area of automatic retopology/quadrangulation.

I stumbled upon a [devtalk.blender.org thread](https://devtalk.blender.org/t/retopology-min-deviation-flow-in-bi-directed-graphs-for-t-mesh-quantization/30763) which mentioned a new algorithm ([QuadWild Bi-MDF](https://github.com/cgg-bern/quadwild-bimdf)) being developed with the intent of its eventual inclusion into Blender.

However, since such an integration has not happened so far and the quadwild library only existed in the form of command-line binaries, I was asked to make a wrapper for it that would make it easier to use from Blender.

This repository/addon is basically that experimental wrapper (with some additional functionality on top). TeamC eventually discontinued this effort due to the remeshing outputs not offering the results they hoped for, but they agreed to have it published as a public repository.

Here's hoping that it comes useful for someone :-)

Unfortunately, there is no documentation at the moment (and I can't predict when I'll get around to write it), but at least now other people have something to play around with.

Still, to get started, here are a few tips:
* The addon adds a "QuadWild" panel in 3D View's sidebar.
* Before you can use it, you'll need to download the quadwild library binaries (press the Install QuadWild button).
* It works in Object mode and operates on selected objects. Once remeshing is done, the original objects will be hidden, and the remeshed objects will be placed in the same location.
* The "Edit flow" functionality (a.k.a. "RoSy combing") is technically working, but seems rarely useful in practice.
* Read the tooltips :-)
