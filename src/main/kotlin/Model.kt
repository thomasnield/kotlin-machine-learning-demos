import javafx.collections.FXCollections
import javafx.scene.paint.Color
import smile.regression.OLS
import java.util.concurrent.ThreadLocalRandom

object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    fun predict(color: Color): FontShade {

        if (inputs.size < 4) return FontShade.LIGHT

        val xInputs = inputs.asSequence()
                .map { it.color.let { doubleArrayOf(it.red, it.green, it.blue) } }
                .toList().toTypedArray()

        val yOutputs = inputs.asSequence()
                .map { it.fontShade.outputValue }
                .toList().toDoubleArray()

        val regression = OLS(xInputs, yOutputs)

        return regression.predict(
                doubleArrayOf(
                        color.red,
                        color.green,
                        color.blue
                )
        ).let { if (it <= 0.0) FontShade.DARK else FontShade.LIGHT }
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
    DARK(Color.BLACK, -1.0),
    LIGHT(Color.WHITE, 1.0)
}

fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)