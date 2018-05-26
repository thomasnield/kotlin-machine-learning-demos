import org.ojalgo.function.aggregator.Aggregator
import org.ojalgo.matrix.BasicMatrix
import tornadofx.*
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.exp

fun main(args: Array<String>) {
/*
    val nn = neuralnetwork {
        inputlayer(nodeCount = 3)
        hiddenlayer(nodeCount =  3)
        outputlayer(nodeCount = 2)

    }

    with(nn) {

        hiddenLayers.forEach {
            it.valuesMatrix.apply(::println)
        }
        nn.outputLayer.valuesMatrix.apply(::println)
    }*/
}

fun neuralnetwork(op: NeuralNetworkBuilder.() -> Unit): NeuralNetwork {
    val nn = NeuralNetworkBuilder()
    nn.op()
    return nn.build().also { it.initialize() }
}

class NeuralNetwork(
        inputNodeCount: Int,
        hiddenLayerCounts: List<Int>,
        outputLayerCount: Int
) {

    val inputLayer = InputLayer(inputNodeCount)

    val hiddenLayers = hiddenLayerCounts.asSequence()
            .map {
                CalculatedLayer(it)
            }.toList().also { layers ->
                layers.withIndex().forEach { (i,layer) ->
                    layer.feedingLayer = (if (i == 0) inputLayer else layers[i-1])
                }
            }

    val outputLayer = CalculatedLayer(outputLayerCount).also {
        it.feedingLayer = (if (hiddenLayers.isNotEmpty()) hiddenLayers.last() else inputLayer)
    }

    fun initialize() {
        hiddenLayers.forEach { it.initializeWeights() }
        outputLayer.initializeWeights()
    }

    fun calculate() {
        hiddenLayers.forEach { it.calculate() }
        outputLayer.calculate()
    }

    fun propogate(errors: DoubleArray) {
        outputLayer.backpropogate(errors)
    }

    private var isInitialized = false

    /**
     * Input a set of training values for each node
     */
    fun trainEntry(inputValues: DoubleArray, desiredOutputs: DoubleArray) {
        if (inputValues.size != inputLayer.count()) throw Exception("There must be ${inputLayer.count()} inputValues")
        if (desiredOutputs.size != outputLayer.count()) throw Exception("There must be ${outputLayer.count()} inputValues")

        // initialize if needed
        if (!isInitialized) {
            initialize()
            isInitialized = true
        }

        // assign input values to input nodes
        inputValues.withIndex().forEach { (i,v) -> inputLayer.nodes[i].value = v }

        // calculate new hidden and output node values
        calculate()

        // propogate error backwards and adjust weights
        val errors = outputLayer.asSequence().map { it.value }.zip(desiredOutputs.asSequence())
                .map { (calculated, desired) ->  calculated - desired }
                .toList().toDoubleArray()


        propogate(errors)
    }

    fun predictEntry(vararg inputValues: Double): DoubleArray {

        if (inputValues.size != inputLayer.count()) throw Exception("There must be ${inputLayer.count()} inputValues")

        // initialize if needed
        if (!isInitialized) {
            initialize()
            isInitialized = true
        }

        // assign input values to input nodes
        inputValues.withIndex().forEach { (i,v) -> inputLayer.nodes[i].value = v }

        // calculate new hidden and output node values
        calculate()
        return outputLayer.map { it.value }.toDoubleArray()
    }
}



// LAYERS
sealed class Layer<N: Node>: Iterable<N> {
    abstract val nodes: List<N>
    override fun iterator() = nodes.iterator()
}

/**
 * An `InputLayer` belongs to the first layer and accepts the input values for each `InputNode`
 */
class InputLayer(nodeCount: Int): Layer<InputNode>() {

    override val nodes = (0..(nodeCount-1)).asSequence()
            .map { InputNode(it) }
            .toList()
}

/**
 * A `CalculatedLayer` is used for the hidden and output layers, and is derived off weights and values off each previous layer
 */
class CalculatedLayer(nodeCount: Int): Layer<CalculatedNode>() {

    var feedingLayer: Layer<out Node> by singleAssign()

    override val nodes by lazy {
        (0..(nodeCount - 1)).asSequence()
                .map { CalculatedNode(it, this) }
                .toList()
    }

    var weightsMatrix: BasicMatrix = primitivematrix(0,0)
    var valuesMatrix: BasicMatrix = primitivematrix(0,0)

    fun initializeWeights() {
        weightsMatrix = primitivematrix(count(), feedingLayer.count()) {
            populate { row,col -> randomInitialValue() }
        }
    }

    fun calculate() {
        valuesMatrix = (weightsMatrix * feedingLayer.toPrimitiveMatrix({it.value})).scalarApply { sigmoid(it) }
    }

    fun backpropogate(errors: DoubleArray) {
        val proportions = weightsMatrix.reduceRows(Aggregator.SUM)
    }
}


// NODES
sealed class Node(val index: Int) {
    abstract val value: Double
}


class InputNode(index: Int): Node(index) {
    override var value = randomInitialValue()
}


class CalculatedNode(index: Int,
                     val parentLayer: CalculatedLayer
                 ): Node(index) {

    override val value get() = parentLayer.valuesMatrix[index.toLong(),0].toDouble()

}

fun randomInitialValue() = ThreadLocalRandom.current().nextDouble(0.0,1.0)
fun sigmoid(x: Number) = 1.0 / (1.0 + exp(-x.toDouble()))

// BUILDERS
class NeuralNetworkBuilder {

    var input = 0
    var hidden = mutableListOf<Int>()
    var output = 0

    fun inputlayer(nodeCount: Int) {
        input = nodeCount
    }

    fun hiddenlayer(nodeCount: Int) {
        hidden.add(nodeCount)
    }

    fun outputlayer(nodeCount: Int) {
        output = nodeCount
    }

    fun build() = NeuralNetwork(input, hidden, output)
}
