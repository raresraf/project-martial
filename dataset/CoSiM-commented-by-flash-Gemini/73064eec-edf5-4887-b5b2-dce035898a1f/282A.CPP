/**
 * @file 282A.CPP
 * @brief Solution to Codeforces Problem 282A: "Bit++".
 *
 * This problem involves a single variable `X`, initially 0. There are `n` operations,
 * each being either `++X`, `X++`, `--X`, or `X--`. The task is to determine the final
 * value of `X` after all operations.
 *
 * Algorithm: Initialize a counter `x` to 0. For each operation string, check its
 * characters to determine if it represents an increment or decrement. Update `x` accordingly.
 *
 * Time Complexity: O(N), where N is the number of operations, as each operation is processed
 *                  in constant time (string input, character access, comparison, and arithmetic).
 * Space Complexity: O(1) for storing variables, as the operation string length is constant.
 */
//#include <bits/stdc++.h> // Commented out: a common competitive programming header for convenience.
//using namespace std;    // Commented out: avoids explicit std:: prefix.
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for std::string.

/**
 * @brief Main function to solve the "Bit++" problem.
 * Sets up fast I/O, reads the number of operations, processes each operation,
 * and prints the final value of the variable `X`.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  std::ios::sync_with_stdio(0); // Disables synchronization with C stdio.
  std::cin.tie(0);              // Unties cin from cout.

  int n;      // Stores the total number of operations.
  int x = 0;  // Stores the current value of the variable X, initialized to 0.
  std::cin >> n; // Reads the number of operations.
  std::string str; // Stores the string representation of the current operation.

  /**
   * Block Logic: Processes each operation to update the value of `x`.
   * Pre-condition: `n` is the total number of operations, `x` is 0.
   * Invariant: `x` reflects the value of the variable after the operations processed so far.
   *
   * Note on loop structure: `for (int i = 0; i < n; n--)` is an unconventional loop.
   * It effectively runs `n` times, decrementing `n` itself in each iteration,
   * while `i` remains unused. It's equivalent to `while (n--) { ... }`.
   */
  for (int i = 0; i < n; n--) {
    std::cin >> str; // Reads the operation string (e.g., "++X", "X--").
    
    // Ternary Operator Logic:
    // Checks if the operation is an increment:
    // `str[0] == '+'` covers "++X".
    // `str[2] == '+'` covers "X++".
    // If either condition is true, `x` is incremented. Otherwise, `x` is decremented.
    (str[0] == '+') || (str[2] == '+') ? x++ : x--;
  }
  
  std::cout << x; // Prints the final value of `x`.

  return 0; // Indicates successful execution.
}
