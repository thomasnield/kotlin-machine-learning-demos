import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import java.util.concurrent.ThreadLocalRandom


object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()
    // if you want to pre-train 1000 observations, uncomment
    /*.apply {

        (1..1000).asSequence().map { randomColor() }.map { CategorizedInput(it, Predictor.FORMULAIC.predictFunction(it)) }
                .forEach { add(it) }
    }*/

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
            (1 - (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue)).let { if (it < .5) FontShade.DARK else FontShade.LIGHT }
        }),

        TOMS_BRUTE_FORCE_NEURAL({ color ->

            fun colorAttributes(c: javafx.scene.paint.Color) = kotlin.doubleArrayOf(
                    c.brightness,
                    c.red,
                    c.green,
                    c.blue
            )

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

fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)