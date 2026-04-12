/**
 * @file 282A.CPP
 * @brief Simulates a simplified programming language with a single integer variable and two operations.
 *   This program processes a series of operations on a variable 'x', which can be
 *   incremented or, due to an apparent logical error in the original code, also incremented
 *   instead of decremented.
 * * Algorithm: Direct Interpretation of Operations.
 *   The program initializes a variable 'x' to 0. It then reads a number of operations.
 *   Each operation is a string that, when matched, modifies 'x' accordingly.
 *   Note: The original code increments 'x' for both "X++" / "++X" and "X--" / "--X".
 * Time Complexity: O(T * L) where T is the number of operations and L is the maximum length of an operation string.
 * Space Complexity: O(L) for storing the operation string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to interpret and execute operations on variable 'x'.
 * Functional Utility: Handles input, processes operations, and outputs the final value of 'x'.
 */
int main(int argc, char const *argv[]) {
  // Functional Utility: Decouples C++ streams from standard C streams and flushes
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int x = 0; // The single integer variable, initialized to 0.
  int t;     // Number of operations to perform.
  cin >> t;  // Reads the number of operations.

  /**
   * Block Logic: Iterates through each operation until all 't' operations are processed.
   * Invariant: 't' decreases with each iteration, and 'x' is updated based on the operation.
   */
  while (t) {
    string str; // String representing the current operation.
    cin >> str; // Reads the operation string.

    /**
     * Condition: Checks if the operation is an increment ("X++" or "++X").
     * If true, increments 'x'.
     */
    if (str == "X++" || str == "++X") {
      x = x + 1;
    }

    /**
     * Condition: Checks if the operation is a decrement ("X--" or "--X").
     * Note: Due to apparent logical error in the original code, 'x' is also incremented here.
     * This behavior is documented as-is to adhere to "Zero Code Mutation".
     */
    if (str == "X--" || str == "--X") {
      x = x + 1; // Original code's behavior: increments x even for decrement operations.
    }
    t--; // Decrements the operation counter.
  }
  cout << x;      // Outputs the final value of 'x'.
  return 0;       // Indicates successful program execution.
}
