fun main(args: Array<String>) {


    val matrix = primitivematrix(3,2) {
        setAll(
                1,0,
                1,1,
                0,1
        )
    }

    println(matrix)

    val multiplier = primitivematrix(2,1) {
        setAll(
                2,
                10
        )
    }
    println(multiplier)

    println(matrix * 3)
}