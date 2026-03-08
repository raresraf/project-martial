/**
 * @file 282A.CPP
 * @brief Solution for a problem involving a series of variable increment/decrement operations.
 *
 * @details
 * This program computes the final value of a variable `x` that starts at 0 and is
 * subjected to a series of operations read from standard input. The operations
 * are either increments ("++X", "X++") or decrements ("--X", "X--"). This
 * implementation uses full string comparison to identify the operation type.
 *
 * @section
 * Algorithm: Iterative accumulation based on string matching.
 * Time Complexity: O(N), where N is the number of operations. Each operation involves a string comparison, but the string length is constant and small.
 * Space Complexity: O(1), as the space for the operation string is reused in each loop iteration.
 */
#include <cstring> // Included but not strictly necessary for this std::string-based implementation.
#include <iostream>
using namespace std;

/**
 * @brief The main function, with an implicit 'int' return type as per C++ standards.
 */
main() {
  // ops: holds the total number of operations.
  // x: the accumulator variable, initialized to zero.
  int ops, x = 0;
  string compute;
  cin >> ops;

  /**
   * @brief This loop processes each of the 'ops' operations.
   *
   * Invariant: On each iteration, the loop reads one operation string and updates 'x'.
   * The loop counter 'ops' is manually decremented at the end of the block.
   */
  while (ops) {
    cin >> compute;
    /**
     * @brief Determines the operation type using full string comparison.
     *
     * Logic: This block explicitly checks if the input string matches one of the
     * increment operations. If it does not, it is assumed to be a decrement.
     * This approach is very readable but slightly less performant than
     * checking a single character.
     */
    if (compute == "X++" || compute == "++X") {
      x++;
    } else {
      x--;
    }
    ops--;
  }
  // Output the final accumulated value of x.
  cout << x;
}
