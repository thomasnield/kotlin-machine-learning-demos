import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.ojalgo.ann.ArtificialNeuralNetwork
import org.ojalgo.array.Primitive64Array
import java.util.concurrent.ThreadLocalRandom

object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    val selectedPredictor = SimpleObjectProperty<Predictor>(Predictor.OJALGO_NN)

    fun predict(color: Color) = selectedPredictor.get().predict(color)

    operator fun plusAssign(categorizedInput: CategorizedInput)  {
        inputs += categorizedInput
    }
    operator fun plusAssign(categorizedInput: Pair<Color,FontShade>)  {
        inputs += categorizedInput.let { CategorizedInput(it.first, it.second) }
    }

    fun preTrainData() {

        PredictorModel::class.java.getResource("color_training_set.csv").readText().lines()
                .asSequence()
                .map { s -> s.split(",").map { it.toInt() } }
                .map { Color.rgb(it[0], it[1], it[2]) }
                .map { CategorizedInput(it, Predictor.FORMULAIC.predict(it))  }
                .toList()
                .forEach {
                    inputs += it
                }
    }


    enum class Predictor {

        /**
         * Uses a simple formula to classify colors as LIGHT or DARK
         */
        FORMULAIC {
            override fun predict(color: Color) = (1 - (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue))
                        .let { if (it < .5) FontShade.DARK else FontShade.LIGHT }
        },
        /**
         * My implementation from scratch, still a work-in-progress
         * I need to implement gradient descent using Koma
         */
        TOMS_FEED_FORWARD_NN {
            override fun predict(color: Color): FontShade {

                val bruteForceNN = neuralnetwork {
                    inputlayer(3)
                    hiddenlayer(3)
                    outputlayer(2)
                }

                val trainingEntries = inputs.asSequence()
                        .map {
                            colorAttributes(it.color) to it.fontShade.outputValue
                        }.asIterable()

                bruteForceNN.trainEntries(trainingEntries)

                val result = bruteForceNN.predictEntry(*colorAttributes(color))
                println("DARK: ${result[0]} LIGHT: ${result[1]}")

                return when {
                    result[0] > result[1] -> FontShade.DARK
                    else -> FontShade.LIGHT
                }
            }
        },
        /**
         * Uses ojAlgo's artificial neural network API
         * Which works really well and is much more lightweight than DL4J
         */
        OJALGO_NN {

            override fun predict(color: Color): FontShade {

                val ann = ArtificialNeuralNetwork.builder(3, 3, 2).apply {

                    activator(0, ArtificialNeuralNetwork.Activator.RECTIFIER)
                    activator(1, ArtificialNeuralNetwork.Activator.SOFTMAX)

                    rate(.05)
                    error(ArtificialNeuralNetwork.Error.CROSS_ENTROPY)

                    val inputValues = inputs.asSequence().map { Primitive64Array.FACTORY.copy(* colorAttributes(it.color)) }
                            .toList()

                    val outputValues = inputs.asSequence().map { Primitive64Array.FACTORY.copy(*it.fontShade.outputValue) }
                            .toList()

                    train(inputValues, outputValues)
                }.get()

                return ann.invoke(Primitive64Array.FACTORY.copy(*colorAttributes(color))).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        /**
         * Uses DeepLearning4J, a heavyweight neural network library that is probably overkill for this toy problem.
         * However, DL4J is a good library to use for large real-world projects.
         */
        DL4J_NN {
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
                        .map { it.fontShade.outputValue }
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

        abstract fun predict(color: Color): FontShade
        override fun toString() = name.replace("_", " ")
    }

}

data class CategorizedInput(
        val color: Color,
        val fontShade: FontShade
)

enum class FontShade(val color: Color, val outputValue: DoubleArray){
    DARK(Color.BLACK, doubleArrayOf(0.0, 1.0)),
    LIGHT(Color.WHITE, doubleArrayOf(1.0,0.0))
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
