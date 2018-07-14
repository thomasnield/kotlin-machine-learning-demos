import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.concurrent.ThreadLocalRandom




object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    val selectedPredictor = SimpleObjectProperty<Predictor>(Predictor.TOMS_FEED_FORWARD_NN)

    fun predict(color: Color) = selectedPredictor.get().predict(color)

    operator fun plusAssign(categorizedInput: CategorizedInput)  {
        inputs += categorizedInput
    }
    operator fun plusAssign(categorizedInput: Pair<Color,FontShade>)  {
        inputs += categorizedInput.let { CategorizedInput(it.first, it.second) }
    }

    fun autoTrain() {

        PredictorModel::class.java.getResource("color_training_set.csv").readText().lines()
                .map { it.split(",").map { it.toInt() } }
                .map { Color.rgb(it[0], it[1], it[2]) }
                .map { CategorizedInput(it, Predictor.FORMULAIC.predict(it))  }
                .forEach {
                    inputs += it
                }
    }


    enum class Predictor {

        FORMULAIC {
            override fun predict(color: Color) = (1 - (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue))
                        .let { if (it < .5) FontShade.DARK else FontShade.LIGHT }
        },

        TOMS_FEED_FORWARD_NN {
            override fun predict(color: Color): FontShade {

                val bruteForceNN = neuralnetwork {
                    inputlayer(4)
                    hiddenlayer(4)
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

        DL4J_NN {
            override fun predict(color: Color): FontShade {

                val dl4jNN = NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(Sgd(.001))
                        .miniBatch(false)
                        .list(
                                DenseLayer.Builder().nIn(4).nOut(4).build(),
                                DenseLayer.Builder().nIn(4).nOut(4).build(),
                                OutputLayer.Builder().nIn(4).nOut(2)
                                        .activation(Activation.SIGMOID)
                                        .lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                                        .build()
                        ).backprop(true)
                        .build()
                        .let(::MultiLayerNetwork).apply {
                            iterationCount = 10000
                            init()
                        }

                val examples = inputs.asSequence()
                        .map { colorAttributes(it.color) }
                        .toList().toTypedArray()
                        .let { Nd4j.create(it) }

                val outcomes = inputs.asSequence()
                        .map { it.fontShade.outputValue }
                        .toList().toTypedArray()
                        .let { Nd4j.create(it) }

                dl4jNN.fit(examples, outcomes)

                val result = dl4jNN.output(Nd4j.create(colorAttributes(color))).toDoubleVector()

                println(result.joinToString(",  "))

                return when {
                    result[0] > result[1] -> FontShade.DARK
                    else -> FontShade.LIGHT
                }
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
        c.brightness,
        c.red,
        c.green,
        c.blue
)