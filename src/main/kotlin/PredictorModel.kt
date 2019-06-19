import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import org.apache.commons.math3.distribution.NormalDistribution
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nield.kotlinstatistics.randomFirst
import org.ojalgo.ann.ArtificialNeuralNetwork
import org.ojalgo.array.Primitive64Array
import java.net.URL
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

object PredictorModel {

    val inputs = FXCollections.observableArrayList<LabeledColor>()

    val selectedPredictor = SimpleObjectProperty<Predictor>(Predictor.OJALGO_NEURAL_NETWORK)

    fun predict(color: Color) = selectedPredictor.get().predict(color)

    operator fun plusAssign(labeledColor: LabeledColor)  {
        inputs += labeledColor
        Predictor.values().forEach { it.retrainFlag = true }
    }
    operator fun plusAssign(categorizedInput: Pair<Color,FontShade>)  {
        inputs += categorizedInput.let { LabeledColor(it.first, it.second) }
        Predictor.values().forEach { it.retrainFlag = true }
    }

    fun preTrainData() {

        URL("https://tinyurl.com/y2qmhfsr")
                .readText().split(Regex("\\r?\\n"))
                .asSequence()
                .drop(1)
                .filter { it.isNotBlank() }
                .map { s ->
                    s.split(",").map { it.toInt() }
                }
                .map { Color.rgb(it[0], it[1], it[2]) }
                .map { LabeledColor(it, Predictor.FORMULAIC.predict(it))  }
                .toList()
                .forEach {
                    inputs += it
                }

        Predictor.values().forEach { it.retrainFlag = true }
    }


    enum class Predictor {

        /**
         * Uses a simple formula to classify colors as LIGHT or DARK
         */
        FORMULAIC {
            override fun predict(color: Color) =  (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue)
                        .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
        },

        LINEAR_REGRESSION_HILL_CLIMBING {

            override fun predict(color: Color): FontShade {

                var redWeightCandidate = 0.0
                var greenWeightCandidate = 0.0
                var blueWeightCandidate = 0.0

                var currentLoss = Double.MAX_VALUE

                val normalDistribution = NormalDistribution(0.0, 1.0)

                fun predict(color: Color) =
                        (redWeightCandidate * color.red + greenWeightCandidate * color.green + blueWeightCandidate * color.blue)

                repeat(10000) {

                    val selectedColor = (0..2).asSequence().randomFirst()
                    val adjust = normalDistribution.sample()

                    // make random adjustment to two of the colors
                    when {
                        selectedColor == 0 -> redWeightCandidate += adjust
                        selectedColor == 1 -> greenWeightCandidate += adjust
                        selectedColor == 2 -> blueWeightCandidate += adjust
                    }

                    // Calculate the loss, which is sum of squares
                    val newLoss = inputs.asSequence()
                            .map { (color, fontShade) ->
                                (predict(color) - fontShade.intValue).pow(2)
                            }.sum()

                    // If improvement doesn't happen, undo the move
                    if (newLoss < currentLoss) {
                        currentLoss = newLoss
                    } else {
                        // revert if no improvement happens
                        when {
                            selectedColor == 0 -> redWeightCandidate -= adjust
                            selectedColor == 1 -> greenWeightCandidate -= adjust
                            selectedColor == 2 -> blueWeightCandidate -= adjust
                        }
                    }
                }

                println("${redWeightCandidate}R + ${greenWeightCandidate}G + ${blueWeightCandidate}B")

                val formulasLoss = inputs.asSequence()
                        .map { (color, fontShade) ->
                            ( (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue) - fontShade.intValue).pow(2)
                        }.average()

                println("BEST LOSS: $currentLoss, FORMULA'S LOSS: $formulasLoss \r\n")

                return predict(color)
                        .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
            }
        },

        LOGISTIC_REGRESSION_HILL_CLIMBING {


            var b0 = .01 // constant
            var b1 = .01 // red beta
            var b2 = .01 // green beta
            var b3 = .01 // blue beta


            fun predictProbability(color: Color) = 1.0 / (1 + exp(-(b0 + b1 * color.red + b2 * color.green + b3 * color.blue)))

            // Helpful Resources:
            // StatsQuest on YouTube: https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
            // Brandon Foltz on YouTube: https://www.youtube.com/playlist?list=PLIeGtxpvyG-JmBQ9XoFD4rs-b3hkcX7Uu
            override fun predict(color: Color): FontShade {


                if (retrainFlag) {
                    var bestLikelihood = -10_000_000.0

                    // use hill climbing for optimization
                    val normalDistribution = NormalDistribution(0.0, 1.0)

                    b0 = .01 // constant
                    b1 = .01 // red beta
                    b2 = .01 // green beta
                    b3 = .01 // blue beta

                    // 1 = DARK FONT, 0 = LIGHT FONT

                    repeat(50000) {

                        val selectedBeta = (0..3).asSequence().randomFirst()
                        val adjust = normalDistribution.sample()

                        // make random adjustment to two of the colors
                        when {
                            selectedBeta == 0 -> b0 += adjust
                            selectedBeta == 1 -> b1 += adjust
                            selectedBeta == 2 -> b2 += adjust
                            selectedBeta == 3 -> b3 += adjust
                        }

                        // calculate maximum likelihood
                        val darkEstimates = inputs.asSequence()
                                .filter { it.fontShade == FontShade.DARK }
                                .map { ln(predictProbability(it.color)) }
                                .sum()

                        val lightEstimates = inputs.asSequence()
                                .filter { it.fontShade == FontShade.LIGHT }
                                .map { ln(1 - predictProbability(it.color)) }
                                .sum()

                        val likelihood = darkEstimates + lightEstimates

                        if (bestLikelihood < likelihood) {
                            bestLikelihood = likelihood
                        } else {
                            // revert if no improvement happens
                            when {
                                selectedBeta == 0 -> b0 -= adjust
                                selectedBeta == 1 -> b1 -= adjust
                                selectedBeta == 2 -> b2 -= adjust
                                selectedBeta == 3 -> b3 -= adjust
                            }
                        }
                    }

                    println("1.0 / (1 + exp(-($b0 + $b1*R + $b2*G + $b3*B))")
                    println("BEST LIKELIHOOD: $bestLikelihood")
                    retrainFlag = false
                }

                return predictProbability(color)
                        .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
            }
        },

        DECISION_TREE {

            // Helpful Resources:
            // StatusQuest on YouTube: https://www.youtube.com/watch?v=7VeUPuFGJHk

            override fun predict(color: Color): FontShade {

                class Feature(val name: String, val mapper: (Color) -> Double)

                val features = listOf(
                        Feature("Red") { it.red * 255.0 },
                        Feature("Green") { it.green * 255.0 },
                        Feature("Blue") { it.blue * 255.0 }
                )

                fun giniImpurityForFeature(feature: Feature,
                                           splitValue: Double,
                                           sampleColors: List<LabeledColor>): Double {

                    val darkColorCount = sampleColors.count { feature.mapper(it.color) >= splitValue }.toDouble()
                    val lightColorCount = sampleColors.count { feature.mapper(it.color) < splitValue }.toDouble()
                    val totalColorCount = sampleColors.count().toDouble()

                    return 1.0 - (darkColorCount / totalColorCount  + .0001).pow(2) -
                            (lightColorCount / totalColorCount + .0001).pow(2)
                }

                fun splitContinuousVariable(feature: Feature, sampleColors: List<LabeledColor>): Double? {

                    val featureValues = sampleColors.asSequence().map { feature.mapper(it.color) }.distinct().toList()

                    val bestSplit = featureValues.asSequence().zipWithNext { value1, value2 -> (value1 + value2) / 2.0 }
                            .minBy { giniImpurityForFeature(feature, it, sampleColors) }

                    return bestSplit
                }

                class TreeLeaf(val feature: Feature,
                               val splitValue: Double,
                               val sampleColors: List<LabeledColor>) {

                    val darkColors = sampleColors.filter { it.fontShade == FontShade.DARK }
                    val lightColors = sampleColors.filter { it.fontShade == FontShade.LIGHT }

                    val giniImpurity = giniImpurityForFeature(feature, splitValue, sampleColors)

                    val darkLeaf: TreeLeaf? = buildLeaf(darkColors, this)
                    val lightLeaf: TreeLeaf? = buildLeaf(lightColors, this)

                    private fun buildLeaf(sampleColors: List<LabeledColor>, previousLeaf: TreeLeaf? = null): TreeLeaf? {
                        val (bestFeature, bestSplit) = features.asSequence()
                                .map { feature ->
                                    feature to splitContinuousVariable(feature, sampleColors)
                                }.filter { (_, split) ->
                                    split != null
                                }.minBy { (feature, split) ->
                                    giniImpurityForFeature(feature, split!!, sampleColors)
                                }?: (null to null)

                        return if (previousLeaf == null || (bestFeature != null && giniImpurityForFeature(bestFeature!!, bestSplit!!, sampleColors) < previousLeaf.giniImpurity))
                            TreeLeaf(bestFeature!!, bestSplit!!, sampleColors)
                        else
                            null
                    }

                    fun predict(color: Color): FontShade {

                        val featureValue = feature.mapper(color)

                        return when {
                            featureValue >= splitValue -> when {
                                darkLeaf == null -> (darkColors.count().toDouble() / sampleColors.count().toDouble())
                                        .let { if (it >= .50) FontShade.DARK else FontShade.LIGHT }
                                else -> darkLeaf.predict(color)
                            }
                            else -> when {
                                lightLeaf == null -> (darkColors.count().toDouble() / sampleColors.count().toDouble())
                                        .let { if (it >= .50) FontShade.DARK else FontShade.LIGHT }
                                else -> lightLeaf.predict(color)
                            }
                        }
                    }
                }

                fun buildLeaf(sampleColors: List<LabeledColor>, previousLeaf: TreeLeaf? = null): TreeLeaf? {
                    val (bestFeature, bestSplit) = features.asSequence()
                            .map { feature ->
                                feature to splitContinuousVariable(feature, sampleColors)
                            }.filter { (_, split) ->
                                split != null
                            }.minBy { (feature, split) ->
                                giniImpurityForFeature(feature, split!!, sampleColors)
                            }!!

                    return if (previousLeaf == null || giniImpurityForFeature(bestFeature, bestSplit!!, sampleColors) < previousLeaf.giniImpurity)
                        TreeLeaf(bestFeature, bestSplit!!, sampleColors)
                    else
                        null
                }


                val tree = buildLeaf(inputs)

                return tree!!.predict(color)
            }
        },

        NEURAL_NETWORK_HILL_CLIMBING {

            lateinit var artificialNeuralNetwork: NeuralNetwork

            override fun predict(color: Color): FontShade {

                if (retrainFlag) {
                    artificialNeuralNetwork = neuralnetwork {
                        inputlayer(3)
                        hiddenlayer(3, ActivationFunction.TANH)
                        outputlayer(2, ActivationFunction.SOFTMAX)
                    }

                    val trainingData = inputs.map { colorAttributes(it.color) to it.fontShade.outputArray }

                    artificialNeuralNetwork.trainEntriesHillClimbing(trainingData)
                    retrainFlag = false
                }
                return artificialNeuralNetwork.predictEntry(colorAttributes(color)).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        NEURAL_NETWORK_SIMULATED_ANNEALING {

            lateinit var artificialNeuralNetwork: NeuralNetwork

            override fun predict(color: Color): FontShade {

                if (retrainFlag) {
                    artificialNeuralNetwork = neuralnetwork {
                        inputlayer(3)
                        hiddenlayer(3, ActivationFunction.TANH)
                        outputlayer(2, ActivationFunction.SOFTMAX)
                    }

                    val trainingData = inputs.map { colorAttributes(it.color) to it.fontShade.outputArray }

                    artificialNeuralNetwork.trainEntriesSimulatedAnnealing(trainingData)
                    retrainFlag = false
                }
                return artificialNeuralNetwork.predictEntry(colorAttributes(color)).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        OJALGO_NEURAL_NETWORK {

            lateinit var artificialNeuralNetwork: ArtificialNeuralNetwork

            override fun predict(color: Color): FontShade {

                if (retrainFlag) {
                    artificialNeuralNetwork = ArtificialNeuralNetwork.builder(3, 3, 2).apply {

                        activator(0, ArtificialNeuralNetwork.Activator.RECTIFIER)
                        activator(1, ArtificialNeuralNetwork.Activator.SOFTMAX)

                        rate(.05)
                        error(ArtificialNeuralNetwork.Error.CROSS_ENTROPY)

                        val inputValues = inputs.asSequence().map { Primitive64Array.FACTORY.copy(* colorAttributes(it.color)) }
                                .toList()

                        val outputValues = inputs.asSequence().map { Primitive64Array.FACTORY.copy(*it.fontShade.outputArray) }
                                .toList()

                        train(inputValues, outputValues)
                    }.get()

                    retrainFlag = false
                }

                return artificialNeuralNetwork.invoke(Primitive64Array.FACTORY.copy(*colorAttributes(color))).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        /**
         * Uses DeepLearning4J, a heavyweight neural network library that is probably overkill for this toy problem.
         * However, DL4J is a good library to use for large real-world projects.
         */
        DL4J_NEURAL_NETWORK {
            override fun predict(color: Color): FontShade {

                val dl4jNN = NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.UNIFORM)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(Nesterovs(.006, .9))
                        .l2(1e-4)
                        .list(
                                DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.RELU).build(),
                                OutputLayer.Builder().nIn(3).nOut(2).activation(Activation.SOFTMAX).build()
                        ).pretrain(false)
                        .backprop(true)
                        .build()
                        .let(::MultiLayerNetwork).apply { init() }

                val examples = inputs.asSequence()
                        .map { colorAttributes(it.color) }
                        .toList().toTypedArray()
                        .let { Nd4j.create(it) }

                val outcomes = inputs.asSequence()
                        .map { it.fontShade.outputArray }
                        .toList().toTypedArray()
                        .let { Nd4j.create(it) }


                // train for 1000 iterations (epochs)
                repeat(1000) {
                    dl4jNN.fit(examples, outcomes)
                }

                // Test the input color and predict it as LIGHT or DARK
                val result = dl4jNN.output(Nd4j.create(colorAttributes(color))).toDoubleVector()

                println(result.joinToString(",  "))

                return if (result[0] > result[1]) FontShade.LIGHT else FontShade.DARK

            }
        };

        var retrainFlag = true

        abstract fun predict(color: Color): FontShade
        override fun toString() = name.replace("_", " ")
    }

}

data class LabeledColor(
        val color: Color,
        val fontShade: FontShade
)

enum class FontShade(val color: Color, val intValue: Double, val outputArray: DoubleArray){
    DARK(Color.BLACK, 1.0, doubleArrayOf(0.0, 1.0)),
    LIGHT(Color.WHITE, 0.0, doubleArrayOf(1.0,0.0))
}

// UTILITIES

fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)


fun randomColor() = (1..3).asSequence()
        .map { randomInt(0,255) }
        .toList()
        .let { Color.rgb(it[0], it[1], it[2]) }

fun colorAttributes(c: Color) = doubleArrayOf(
        c.red,
        c.green,
        c.blue
)
