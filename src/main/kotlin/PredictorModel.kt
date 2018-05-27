import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import java.util.concurrent.ThreadLocalRandom
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.nd4j.linalg.learning.config.Sgd
import org.deeplearning4j.nn.conf.layers.*

object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    val selectedPredictor = SimpleObjectProperty<Predictor>(Predictor.TOMS_BRUTE_FORCE_NEURAL)

    val nn = neuralnetwork {
        inputlayer(4)
        hiddenlayer(4)
        outputlayer(2)
    }

    fun predict(color: Color) = selectedPredictor.get().predictFunction(color)


    operator fun plusAssign(categorizedInput: CategorizedInput)  {
        inputs += categorizedInput
    }
    operator fun plusAssign(categorizedInput: Pair<Color,FontShade>)  {
        inputs += categorizedInput.let { CategorizedInput(it.first, it.second) }
    }


    enum class Predictor(val predictFunction: (Color) -> FontShade) {

        FORMULAIC({ color ->
            (1 - (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue))
                    .let { if (it < .5) FontShade.DARK else FontShade.LIGHT }
        }),

        TOMS_BRUTE_FORCE_NEURAL({ color ->

            val trainingEntries = inputs.asSequence()
                    .map {
                        colorAttributes(it.color) to kotlin.doubleArrayOf(it.fontShade.outputValue)
                    }.asIterable()

            nn.trainEntries(trainingEntries)

            val result = nn.predictEntry(*colorAttributes(color))
            kotlin.io.println("DARK: ${result[0]} LIGHT: ${result[1]}")

            when {
                result[0] > result[1] -> FontShade.DARK
                else -> FontShade.LIGHT
            }
        }),

        DL4J({ color ->
            val conf = NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SIGMOID)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Sgd(.05))
                    .list(
                        DenseLayer.Builder().nIn(4).nOut(4).build(),
                            OutputLayer.Builder().activation(Activation.SIGMOID).nIn(4).nOut(2).build()
                    ).backprop(true)
                    .build()


            FontShade.DARK
        });

        override fun toString() = name.replace("_", " ")
    }

}

data class CategorizedInput(
        val color: Color,
        val fontShade: FontShade
)

enum class FontShade(val color: Color, val outputValue: Double){
    DARK(Color.BLACK, 0.0),
    LIGHT(Color.WHITE, 1.0)
}

// UTILITIES

fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)


fun randomColor() = (1..3).asSequence()
        .map { randomInt(0,255) }
        .toList()
        .let { Color.rgb(it[0], it[1], it[2]) }

fun colorAttributes(c: javafx.scene.paint.Color) = kotlin.doubleArrayOf(
        c.brightness,
        c.red,
        c.green,
        c.blue
)