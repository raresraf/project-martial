/**
 * @file 282A.CPP
 * @brief This program solves the "Bit++" problem (Codeforces 282A).
 *   It simulates a simple programming language with a single variable `X` and operations
 *   to increment (`++X` or `X++`) or decrement (`--X` or `X--`) its value.
 *   The task is to calculate the final value of `X` after `n` operations, starting with `X=0`.
 *   Problem description: https://codeforces.com/problemset/problem/282/A
 * * Algorithm: Iterative sum.
 *   The program reads `n` operation strings. For each string, it determines if it's an
 *   increment operation. If so, it increments a counter; otherwise, it decrements the counter.
 * Time Complexity: O(N * L), where N is the number of operations and L is the length of
 *   an operation string. Since L is a small constant (typically 3), it is effectively O(N).
 * Space Complexity: O(L) to store the operation string `str`.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
using namespace std;

/**
 * @brief Main function to calculate the final value of variable X.
 * Functional Utility: Reads the number of operations and each operation string,
 *                     then computes and prints the final value of X.
 */
int main() {
  int n;       // Stores the number of operations.
  int ans = 0; // Stores the current value of the variable X, initialized to 0.
  string str;  // Stores the operation string read from input.

  // Block Logic: Reads the total number of operations, `n`.
  cin >> n;
  /**
   * Block Logic: Iterates `n` times to process each operation.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `ans` correctly reflects the value of X after `i` operations.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the current operation string.
    cin >> str;
    /**
     * Conditional Logic: Checks if the operation is an increment.
     * `str == "X++" || str == "++X"` checks for both forms of increment operation.
     * If it's an increment, `ans` is increased. Otherwise (it must be a decrement), `ans` is decreased.
     */
    if (str == "X++" || str == "++X")
      ans++; // Increment X.
    else
      ans--; // Decrement X.
  }

  // Block Logic: Prints the final calculated value of X.
  cout << ans << endl;
  return 0;
}
