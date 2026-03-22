/**
 * @file 282A.CPP
 * @brief Simulates a sequence of increment/decrement operations.
 *
 * This program models a scenario where a variable 'x' is modified based on a series of
 * string commands. It serves as a basic example of direct simulation and string parsing.
 *
 * @b Algorithm: Direct Simulation
 * @b Time_Complexity: O(N), where N is the number of operations.
 * @b Space_Complexity: O(1), ignoring input string storage.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int num;
  // 'x' accumulates the result of the operations.
  int x = 0;
  cin >> num;

  /**
   * @brief Main simulation loop.
   * Pre-condition: `num` contains the total number of operations to process.
   * Invariant: `x` holds the net result of the first `i` operations.
   */
  for (int i = 0; i < num; i++) {
    string X;
    cin >> X;
    /**
     * @brief Parses the operation string to determine whether to increment or decrement.
     * Logic: Both "++X" and "X++" result in an increment, while "--X" and "X--" result in a decrement.
     */
    if (X.compare("X++") == 0 || X.compare("++X") == 0) {
      x++;
    } else if (X.compare("X--") == 0 || X.compare("--X") == 0) {
      x--;
    }
  }
  
  // Output the final state of the variable 'x'.
  cout << x;
  return 0;
}
