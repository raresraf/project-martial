/**
 * @file 282A.CPP
 * @brief Solution for a simple programming problem involving variable manipulation.
 * @details This program simulates a basic programming language with a single variable 'x' and four operations.
 * It reads an integer 'n', which is the number of operations to follow. Then, it iterates 'n' times,
 * reading each operation and updating the value of 'x' accordingly.
 * The operations "X++" and "++X" increment 'x', while "X--" and "--X" decrement 'x'.
 * Finally, it prints the final value of 'x'.
 *
 * @algorithm Simple counter simulation
 * @complexity
 * Time: O(N), where N is the number of operations, because each operation is processed in constant time.
 * Space: O(1), as the program uses a fixed amount of memory regardless of the input size.
 */
#include <iostream>
#include <string> // Include for std::string usage

using namespace std;

int main() {
  // n: the number of operations.
  // x: the variable to be manipulated, initialized to 0.
  int n = 0, x = 0;
  // op: stores the operation string.
  string op = "";

  // Read the total number of operations.
  cin >> n;

  /**
   * @brief Loop to process each operation.
   * This loop runs 'n' times, once for each operation.
   */
  for (int i = 0; i < n; i++) {
    // Read the operation string.
    cin >> op;

    /**
     * @brief Conditional block to update 'x' based on the operation.
     * Pre-condition: 'op' contains one of the four valid operation strings.
     * Invariant: 'x' is incremented if the operation contains "++", and decremented if it contains "--".
     * This could also be implemented by checking op[1] == '+' or op[1] == '-'.
     */
    if (op == "X++" || op == "++X") {
      x++;
    } else if (op == "X--" || op == "--X") {
      x--;
    }
  }

  // Print the final value of x.
  cout << x << endl;

  return 0; // Indicate successful execution.
}