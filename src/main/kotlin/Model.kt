import javafx.collections.FXCollections
import javafx.scene.paint.Color
import java.util.concurrent.ThreadLocalRandom


object PredictorModel {

    val inputs = FXCollections.observableArrayList<CategorizedInput>()

    fun predict(color: Color): FontShade {
        return FontShade.LIGHT
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