# Kotlin Simple Neural Network

This is a simple neural network project that will suggest a LIGHT or DARK font for a given background color with an interactive UI.

The training/predicting user interface was built with [TornadoFX](https://github.com/edvin/tornadofx).

![demo.gif]

Currently, the neural network has been implemented from scratch with some matrix calculation help from [ojAlgo](https://github.com/optimatika/ojAlgo). No backpropagation is currently implemented, and a brute-force random algorithm will find optimal weight values.

Tariq Rashid's book [Build Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/) has been a tremendously helpful resource, as we as [3Blue1Brown's Video](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).

I will document and share more soon, and hopefully get around to implementing proper backpropagation.