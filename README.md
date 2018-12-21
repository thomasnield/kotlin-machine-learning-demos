# Kotlin Simple Neural Network

This is a simple neural network application that will suggest a LIGHT or DARK font for a given background color.

The training/predicting user interface was built with [TornadoFX](https://github.com/edvin/tornadofx).

## YouTube Walkthrough

[![](https://img.youtube.com/vi/tAioWlhKA90/hqdefault.jpg)](https://www.youtube.com/watch?v=tAioWlhKA90)

## Featured at KotlinConf 2018 in Amsterdam

[![](https://img.youtube.com/vi/-zTqtEcnM7A/hqdefault.jpg)](https://youtu.be/-zTqtEcnM7A)

## Details

Currently there are three implementations: 

1) [Simple RGB formula](https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color#1855903) 
2) My feed-forward brute force implementation (no backpropagation)
3) [ojAlgo! Neural Network](http://www.ojalgo.org/)
4) [DeepLearning4J](https://deeplearning4j.org/)

For this simple toy example ojAlgo seems to perform the best, and is light and the simplest to implement. DL4J is definitely more heavyweight (with many dependencies) but is a more robust framework for larger, data-intensive deep learning problems in production. DL4J also has a nice [Kotlin MNIST example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/kotlin/org/deeplearning4j/examples/feedforward/mnist). 

Note also there is now a button to pre-train 1345 categorized colors. 

Tariq Rashid's book [Build Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/) is a tremendous resource, as well as [3Blue1Brown's Video](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) is probably the most thorough and useful resource when you are ready to deep-dive.



