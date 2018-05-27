import javafx.collections.FXCollections
import javafx.scene.paint.Color
import java.util.concurrent.ThreadLocalRandom


object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    val nn = neuralnetwork {
        inputlayer(4)
        hiddenlayer(4)
        outputlayer(2)
    }

    fun predict(color: Color): FontShade {

        fun colorAttributes(c: Color) = doubleArrayOf(
                c.brightness,
                c.red,
                c.green,
                c.blue
        )


        val trainingEntries = inputs.asSequence()
                .map {
                    colorAttributes(it.color) to doubleArrayOf(it.fontShade.outputValue)
                }.asIterable()

        nn.trainEntries(trainingEntries)

        val result = nn.predictEntry(*colorAttributes(color))
        println("DARK: ${result[0]} LIGHT: ${result[1]}")

        return when {
            result[0] > result[1] -> FontShade.DARK
            else -> FontShade.LIGHT
        }
    }


    operator fun plusAssign(categorizedInput: CategorizedInput)  {
        inputs += categorizedInput
    }
    operator fun plusAssign(categorizedInput: Pair<Color,FontShade>)  {
        inputs += categorizedInput.let { CategorizedInput(it.first, it.second) }
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