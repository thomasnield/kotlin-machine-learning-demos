import javafx.application.Application
import javafx.beans.property.ReadOnlyObjectWrapper
import javafx.beans.property.SimpleObjectProperty
import javafx.geometry.Insets
import javafx.geometry.Orientation
import javafx.scene.layout.Background
import javafx.scene.layout.BackgroundFill
import javafx.scene.layout.CornerRadii
import javafx.scene.paint.Color
import javafx.scene.text.FontWeight
import tornadofx.*


fun main(args: Array<String>) = Application.launch(MainApp::class.java, *args)

class MainApp: App(MainView::class)

class MainView: View() {


    val backgroundColor = SimpleObjectProperty(Color.GRAY)

    fun assignRandomColor() = randomColor()
            .also { backgroundColor.set(it) }

    override val root = splitpane {

        title = "Light/Dark Text Suggester"
        orientation = Orientation.HORIZONTAL

        borderpane {

            top = label("TRAIN") {
                style {
                    textFill =  Color.RED
                    fontWeight = FontWeight.BOLD
                }
            }

            center = form {
                fieldset {

                    field("Which looks better?").hbox {
                        button("DARK") {
                            textFill = Color.BLACK
                            useMaxWidth = true

                            backgroundProperty().bind(
                                    backgroundColor.select { ReadOnlyObjectWrapper(Background(BackgroundFill(it, CornerRadii.EMPTY, Insets.EMPTY))) }
                            )

                            setOnAction {

                                PredictorModel += CategorizedInput(backgroundColor.get(), FontShade.DARK)
                                assignRandomColor()
                            }
                        }

                        button("LIGHT") {
                            textFill = Color.WHITE
                            useMaxWidth = true

                            backgroundProperty().bind(
                                    backgroundColor.select { ReadOnlyObjectWrapper(Background(BackgroundFill(it, CornerRadii.EMPTY, Insets.EMPTY))) }
                            )

                            setOnAction {
                                PredictorModel += CategorizedInput(backgroundColor.get(), FontShade.DARK)

                                assignRandomColor()
                            }
                        }
                    }
                }

                fieldset {
                    field("Model") {
                        combobox(PredictorModel.selectedPredictor) {

                            PredictorModel.Predictor.values().forEach { items.add(it) }
                        }
                    }
                }

                fieldset {
                    field("Pre-Train") {
                        button("Train 1345 Colors") {
                            useMaxWidth = true
                            setOnAction {
                                PredictorModel.preTrainData()
                                isDisable = true
                            }
                        }
                    }
                }
            }

        }

        borderpane {

            top = label("PREDICT") {
                style {
                    textFill =  Color.RED
                    fontWeight = FontWeight.BOLD
                }
            }

            center = form {
                fieldset {
                    field("Background") {
                        colorpicker {
                            valueProperty().onChange {
                                backgroundColor.set(it)
                            }

                            customColors.forEach { println(it) }
                        }
                    }
                    field("Result") {
                        label("LOREM IPSUM") {
                            backgroundProperty().bind(
                                    backgroundColor.select { ReadOnlyObjectWrapper(Background(BackgroundFill(it, CornerRadii.EMPTY, Insets.EMPTY))) }
                            )

                            backgroundColor.onChange {
                                val result = PredictorModel.predict(it!!)

                                text = result.toString()
                                textFill = result.color
                            }

                        }
                    }
                }
            }
        }
    }
}
