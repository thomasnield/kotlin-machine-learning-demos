import org.apache.commons.math3.distribution.TDistribution
import org.nield.kotlinstatistics.randomFirst
import org.nield.kotlinstatistics.weightedCoinFlip
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
    fun trainEntriesHillClimbing(inputsAndTargets: Iterable<Pair<DoubleArray, DoubleArray>>) {

        val entries = inputsAndTargets.toList()


        // use simple hill climbing
        var bestLoss = Double.MAX_VALUE

        val tDistribution = TDistribution(3.0)

        val allCalculatedNodes = calculatedLayers.asSequence().flatMap {
            it.nodes.asSequence()
        }.toList()

        println("Training with ${entries.count()}")

        val learningRate = .5

        repeat(100_000) { epoch ->

            val randomlySelectedNode = allCalculatedNodes.randomFirst()
            val randomlySelectedFeedingNode = randomlySelectedNode.layer.feedingLayer.nodes.randomFirst()
            val selectedWeightKey = WeightKey(randomlySelectedNode.layer.index, randomlySelectedFeedingNode.index, randomlySelectedNode.index)

            val currentWeightValue = randomlySelectedNode.layer.weights[selectedWeightKey]
                    ?: throw Exception("$selectedWeightKey not found in ${randomlySelectedNode.layer.weights}")

            val randomAdjust = tDistribution.sample().let { it * learningRate }.let {
                when {
                    currentWeightValue + it < -1.0 -> -1.0 - currentWeightValue
                    currentWeightValue + it > 1.0 -> 1.0 - currentWeightValue
                    else -> it
                }
            }

            randomlySelectedNode.layer.modifyWeight(selectedWeightKey, randomAdjust)

            val totalLoss = entries
                    .asSequence()
                    .flatMap { (input,label) ->
                      label.asSequence()
                                .zip(predictEntry(input).asSequence()) { actual, predicted -> (actual-predicted).pow(2) }
                    }.sum()

            if (totalLoss < bestLoss) {
                println("epoch $epoch: $bestLoss -> $totalLoss")
                bestLoss = totalLoss
            } else {
                randomlySelectedNode.layer.modifyWeight(selectedWeightKey, -randomAdjust)
            }
        }

        calculatedLayers.forEach { println(it.weights) }
    }

    fun trainEntriesSimulatedAnnealing(inputsAndTargets: Iterable<Pair<DoubleArray, DoubleArray>>) {

        val entries = inputsAndTargets.toList()


        // use simulated annealing
        var bestLoss = Double.MAX_VALUE
        var currentLoss = bestLoss
        var bestConfig = calculatedLayers.map { it.index to it.weights.toMap() }.toMap()

        val tDistribution = TDistribution(3.0)

        val allCalculatedNodes = calculatedLayers.asSequence().flatMap {
            it.nodes.asSequence()
        }.toList()

        println("Training with ${entries.count()}")

        val learningRate = .5

        sequenceOf(
                generateSequence(80.0) { t -> t - .0001 }.takeWhile { it >= 0 },
                generateSequence(120.0) { t -> t - .0001 }.takeWhile { it >= 0 }
        ).flatMap { it }.forEach { temp ->

            val randomlySelectedNode = allCalculatedNodes.randomFirst()
            val randomlySelectedFeedingNode = randomlySelectedNode.layer.feedingLayer.nodes.randomFirst()
            val selectedWeightKey = WeightKey(randomlySelectedNode.layer.index, randomlySelectedFeedingNode.index, randomlySelectedNode.index)

            val currentWeightValue = randomlySelectedNode.layer.weights[selectedWeightKey]
                    ?: throw Exception("$selectedWeightKey not found in ${randomlySelectedNode.layer.weights}")

            val randomAdjust = tDistribution.sample().let { it * learningRate }.let {
                when {
                    currentWeightValue + it < -1.0 -> -1.0 - currentWeightValue
                    currentWeightValue + it > 1.0 -> 1.0 - currentWeightValue
                    else -> it
                }
            }

            randomlySelectedNode.layer.modifyWeight(selectedWeightKey, randomAdjust)

            val newLoss = entries
                    .asSequence()
                    .flatMap { (input,label) ->
                        label.asSequence()
                                .zip(predictEntry(input).asSequence()) { actual, predicted -> (actual-predicted).pow(2) }
                    }.sum()

            if (newLoss < currentLoss) {

                currentLoss = newLoss

                if (newLoss < bestLoss) {
                    println("temp $temp: $bestLoss -> $newLoss")
                    bestLoss = newLoss
                    bestConfig = calculatedLayers.asSequence().map { it.index to it.weights.toMap() }.toMap()
                }
            } else if (weightedCoinFlip(exp((-(newLoss - currentLoss) ) / temp))) {
                //println("temp $temp: $newLoss <- $bestLoss")
                currentLoss = newLoss
            } else {
                randomlySelectedNode.layer.modifyWeight(selectedWeightKey, -randomAdjust)
            }
        }

        calculatedLayers.forEach { cl -> bestConfig[cl.index]!!.forEach { w -> cl.weights.set(w.key, w.value) }}
        calculatedLayers.forEach { println(it.weights) }
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
                                WeightKey(index, feedingNodeIndex, nodeIndex) to randomWeightValue()
                            }
                }.toMap().toMutableMap()
    }

    fun modifyWeight(key: WeightKey,  adjustment: Double) =
            weights.compute(key) { k, v -> v!! + adjustment }
}


// NODES
sealed class Node(val index: Int) {
    abstract val value: Double
}

class InputNode(index: Int): Node(index) {
    override var value = 0.0
}


class CalculatedNode(index: Int, val layer: CalculatedLayer): Node(index) {

    override val value: Double get() = layer.feedingLayer.asSequence()
            .map { feedingNode ->
                val weightKey = WeightKey(layer.index, feedingNode.index, index)
                layer.weights[weightKey]!! * feedingNode.value
            }.sum()
            .let { v ->

                layer.activationFunction.invoke(v) {
                    layer.asSequence().map { node ->
                        node.layer.feedingLayer.asSequence()
                                .map { feedingNode ->
                                    val weightKey = WeightKey(layer.index, feedingNode.index, node.index)
                                    layer.weights[weightKey]!! * feedingNode.value
                                }.sum()
                    }.toList().toDoubleArray()
                }
            }
}

fun randomWeightValue() = ThreadLocalRandom.current().nextDouble(-1.0,1.0)

enum class ActivationFunction {

    IDENTITY {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) =  x
    },
    SIGMOID {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) =  1.0 / (1.0 + exp(-x))
    },
    TANH {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = kotlin.math.tanh(x)
    },
    RELU {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = if (x < 0.0) 0.0 else x
    },
    MAX {
        override fun invoke (x: Double, otherValues: () -> DoubleArray) = if (x == otherValues().max()) x else 0.0
    },
    SOFTMAX {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) =
                (exp(x) / otherValues().asSequence().map { exp(it) }.sum())
                        /*.also { println("${exp(x)} / ${(otherValues().asSequence().map { exp(it) }.sum())} = $it") }*/
    };

    abstract fun invoke(x: Double, otherValues: () -> DoubleArray): Double
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