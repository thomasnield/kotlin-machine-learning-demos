import java.util.concurrent.ThreadLocalRandom

fun main(args: Array<String>) {


    neuralnetwork {

        inputlayer(nodeCount = 3)


    }
}

fun neuralnetwork(op: NeuralNetworkBuilder.() -> Unit): NeuralNetworkBuilder {
    val nn = NeuralNetworkBuilder()
    nn.op()
    return nn
}

class NeuralNetwork(
        val inputLayer: Layer,
        val hiddenLayers: List<Layer>,
        val outputLayer: Layer
)

class NeuralNetworkBuilder {

    val input = mutableListOf<Node>()
    val hidden = mutableListOf<MutableList<Node>>()
    val output = mutableListOf<Node>()


    fun inputlayer(nodeCount: Int) = inputlayer {
        nodecount(nodeCount)
    }

    fun inputlayer(op: LayerBuilder.() -> Unit) {
        LayerBuilder(input).op()
    }

    fun hiddenlayer(op: LayerBuilder.() -> Unit) {
        val newHiddenLayer = mutableListOf<Node>()
        LayerBuilder(newHiddenLayer).op()
        hidden += newHiddenLayer
    }
    fun hiddenlayer(nodeCount: Int) = hiddenlayer {
        nodecount(nodeCount)
    }

    fun outputlayer(op: LayerBuilder.() -> Unit) {
        LayerBuilder(output).op()
    }

    fun outputlayer(nodeCount: Int) = outputlayer {
        nodecount(nodeCount)
    }

    fun build() = NeuralNetwork(
                inputLayer = Layer(input),
                hiddenLayers = hidden.asSequence().withIndex()
                        .map { (i,list) ->
                            if (i == 0) Layer(list, input) else Layer(list, hidden[i-1])
                        }.toList(),
                outputLayer = Layer(output, hidden.last())
                )
}
class Layer(val nodes: List<Node>, val previousNodes: List<Node>? = null) {

    val valueMatrix get() = nodes.toPrimitiveMatrix({it.value})

    //val weightMatrix get() =

}
class LayerBuilder(val nodes: MutableList<Node>) {
    var indexer = 0

    fun node() {
        nodes += Node(indexer++)
    }

    fun nodecount(ct: Int) = (1..ct).forEach { node() }
}

class Node(val index: Int,
           var value: Double =
                   ThreadLocalRandom.current().nextDouble(0.0,1.0)
)



