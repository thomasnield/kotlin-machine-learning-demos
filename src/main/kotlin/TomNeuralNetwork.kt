import org.apache.commons.math3.distribution.TDistribution
import org.nield.kotlinstatistics.random
import org.nield.kotlinstatistics.randomDistinct
import org.nield.kotlinstatistics.randomFirst
import tornadofx.singleAssign
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.exp
import kotlin.math.pow

fun neuralnetwork(op: NeuralNetworkBuilder.() -> Unit): NeuralNetwork {
    val nn = NeuralNetworkBuilder()
    nn.op()
    return nn.build()
}

class NeuralNetwork(
        inputNodeCount: Int,
        hiddenLayers: List<NeuralNetworkBuilder.HiddenLayerBuilder>,
        outputLayer: NeuralNetworkBuilder.HiddenLayerBuilder
) {


    val inputLayer = InputLayer(inputNodeCount)

    val hiddenLayers = hiddenLayers.asSequence()
            .mapIndexed { index,hiddenLayer ->
                CalculatedLayer(index, hiddenLayer.nodeCount, hiddenLayer.activationFunction)
            }.toList().also { layers ->
                layers.withIndex().forEach { (i,layer) ->
                    layer.feedingLayer = (if (i == 0) inputLayer else layers[i-1])
                }
            }

    val outputLayer = CalculatedLayer(hiddenLayers.count(), outputLayer.nodeCount, outputLayer.activationFunction).also {
        it.feedingLayer = (if (this.hiddenLayers.isNotEmpty()) this.hiddenLayers.last() else inputLayer)
    }

    val calculatedLayers = this.hiddenLayers.plusElement(this.outputLayer)


    /**
     * Input a set of training values for each node
     */
    fun trainEntries(inputsAndTargets: Iterable<Pair<DoubleArray, DoubleArray>>) {

        val entries = inputsAndTargets.toList()


        // use simple hill climbing
        var bestLoss = Double.MAX_VALUE

        val tDistribution = TDistribution(3.0)

        val allCalculatedNodes = calculatedLayers.asSequence().flatMap {
            it.nodes.asSequence()
        }.toList()

        repeat(100_000) {

            val randomlySelectedNode = allCalculatedNodes.randomFirst()
            val randomlySelectedFeedingNode = randomlySelectedNode.layer.feedingLayer.nodes.randomFirst()
            val randomlySelectedWeightKey = WeightKey(randomlySelectedNode.layer.index, randomlySelectedNode.index, randomlySelectedFeedingNode.index)

            val randomAdjust = tDistribution.sample()

            randomlySelectedNode.layer.modifyWeight(randomlySelectedWeightKey, randomAdjust)

            val totalLoss = entries.asSequence()
                    .flatMap {
                        it.second.asSequence()
                                .zip(predictEntry(it.first).asSequence()) { actual, predicted -> (actual-predicted).pow(2) }
                    }.average()

            if (totalLoss < bestLoss) {
                //println("$bestLoss -> $totalLoss")
                bestLoss = totalLoss
            } else {
                randomlySelectedNode.layer.modifyWeight(randomlySelectedWeightKey, -randomAdjust)
            }
        }
    }

    fun predictEntry(inputValues: DoubleArray): DoubleArray {


        // assign input values to input nodes
        inputValues.withIndex().forEach { (i,v) -> inputLayer.nodes[i].value = v }

        // calculate new hidden and output node values
        return outputLayer.map { it.value }.toDoubleArray()
    }
}


data class WeightKey(val calculatedLayerIndex: Int, val feedingNodeIndex: Int, val nodeIndex: Int)



// LAYERS
sealed class Layer<N: Node>: Iterable<N> {
    abstract val nodes: List<N>
    override fun iterator() = nodes.iterator()
}

/**
 * An `InputLayer` belongs to the first layer and accepts the input values for each `InputNode`
 */
class InputLayer(nodeCount: Int): Layer<InputNode>() {

    override val nodes = (0 until nodeCount).asSequence()
            .map { InputNode(it) }
            .toList()
}

/**
 * A `CalculatedLayer` is used for the hidden and output layers, and is derived off weights and values off each previous layer
 */
class CalculatedLayer(val index: Int, nodeCount: Int, val activationFunction: ActivationFunction): Layer<CalculatedNode>() {

    var feedingLayer: Layer<out Node> by singleAssign()

    override val nodes by lazy {
        (0 until nodeCount).asSequence()
                .map { CalculatedNode(it, this) }
                .toList()
    }

    // weights are paired for feeding layer and this layer
    val weights by lazy {
        (0 until feedingLayer.nodes.count())
                .asSequence()
                .flatMap { feedingNodeIndex ->
                    (0 until nodeCount).asSequence()
                            .map { nodeIndex ->
                                WeightKey(index, feedingNodeIndex, nodeIndex) to 0.0
                            }
                }.toMap().toMutableMap()
    }

    fun modifyWeight(key: WeightKey,  adjustment: Double) =
            weights.compute(key) { k, v ->
                ((v ?: 0.0) + adjustment).let {
                    when {
                        it < -1.0 -> -1.0
                        it > 1.0 -> 1.0
                        else -> it
                    }
                }
            }
}


// NODES
sealed class Node(val index: Int) {
    abstract val value: Double
}

class InputNode(index: Int): Node(index) {
    override var value = randomWeightValue()
}


class CalculatedNode(index: Int, val layer: CalculatedLayer
): Node(index) {

    override val value get() = layer.feedingLayer.asSequence()
            .map { feedingNode ->
                val weightKey = WeightKey(layer.index, feedingNode.index, index)

                layer.weights[weightKey]!! * feedingNode.value }
            .sum()
            .let { layer.activationFunction.invoke(it, layer) }
}

fun randomWeightValue() = ThreadLocalRandom.current().nextDouble(-1.0,1.0)

enum class ActivationFunction {
    SIGMOID {
        override fun invoke(x: Double, calculatedLayer: CalculatedLayer) =  1.0 / (1.0 + exp(-x))
    },
    RELU {
        override fun invoke(x: Double, calculatedLayer: CalculatedLayer) = if (x < 0.0) 0.0 else x
    },
    SOFTMAX {
        override fun invoke(x: Double, calculatedLayer: CalculatedLayer) = exp(x) / calculatedLayer.nodes.asSequence().map { it.value }.map { exp(it) }.sum()
    };

    abstract fun invoke(x: Double, calculatedLayer: CalculatedLayer): Double
}

// BUILDERS
class NeuralNetworkBuilder {

    var input = 0
    var hidden = mutableListOf<HiddenLayerBuilder>()
    var output: HiddenLayerBuilder = HiddenLayerBuilder(0,ActivationFunction.RELU)

    class HiddenLayerBuilder(val nodeCount: Int, val activationFunction: ActivationFunction)

    fun inputlayer(nodeCount: Int) {
        input = nodeCount
    }

    fun hiddenlayer(nodeCount: Int, activationFunction: ActivationFunction) {
        hidden.add(HiddenLayerBuilder(nodeCount,activationFunction))
    }

    fun outputlayer(nodeCount: Int, activationFunction: ActivationFunction) {
        output = HiddenLayerBuilder(nodeCount,activationFunction)
    }

    fun build() = NeuralNetwork(input, hidden, output)
}