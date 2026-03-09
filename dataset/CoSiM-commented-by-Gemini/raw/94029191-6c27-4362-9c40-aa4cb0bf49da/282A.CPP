/**
 * @file 282A.CPP
 * @brief Executes a series of simple increment/decrement operations.
 *
 * This program solves the "Bit++" problem from Codeforces. It simulates a
 * simple programming language with one variable, `x`, initialized to 0. It
 * processes a series of operations (`++X`, `X++`, `--X`, `X--`) and prints the
 * final value of `x`.
 *
 * @algorithm String Parsing and Simulation
 * @complexity
 * - Time: O(N), where N is the number of operations, because it iterates through
 *   the operations once.
 * - Space: O(1), aside from the storage for the input string, which is reused.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of operations to perform.
  // x: the single variable, initialized to 0.
  int n, x(0);
  cin >> n;

  string s;

  /**
   * @brief Block Logic: Reads and executes each of the `n` operations.
   * Invariant: After each iteration, `x` holds the accumulated result of all
   *            operations processed so far.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Logic: Determines the operation by checking the middle character.
     * This is a clever simplification: for all valid inputs ("++X", "X++", "--X", "X--"),
     * the character at index 1 is always '+' for an increment and '-' for a decrement.
     */
    if (s[1] == '+') {
      ++x;
    } else {
      --x;
    }
  }

  // Output the final value of x.
  cout << x << endl;
  return 0;
}
