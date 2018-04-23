import javafx.scene.paint.Color
import java.util.concurrent.ThreadLocalRandom

object PredictorModel {

    fun forBackgroundColor(color: Color) = if (randomInt(0,1) == 0) FontShade.DARK else FontShade.LIGHT// returns predicted color
}

enum class FontShade(val color: Color){
    DARK(Color.BLACK),
    LIGHT(Color.WHITE)
}

fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)