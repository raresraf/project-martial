/**
 * @file 282A.CPP
 * @brief This program solves the "Bit++" problem (Codeforces 282A).
 *   It simulates a simple programming language with a single variable `X` and operations
 *   to increment (`++X` or `X++`) or decrement (`--X` or `X--`) its value.
 *   The task is to calculate the final value of `X` after `n` operations, starting with `X=0`.
 *   Problem description: https://codeforces.com/problemset/problem/282/A
 * * Algorithm: Iterative sum with regex-based operation detection.
 *   1. Reads `n` (number of operations).
 *   2. For each operation string, uses `std::regex_match` to check for the presence of "--"
 *      to identify decrement operations.
 *   3. Increments or decrements a counter (`X`'s value) accordingly.
 * Time Complexity: O(N * (L + R)), where N is the number of operations, L is the average length
 *   of an operation string, and R is the complexity of regex matching. `std::regex_match` can
 *   be computationally more expensive than simple string comparisons.
 * Space Complexity: O(L) for storing the operation string, plus additional space for the regex engine.
 */
#include <iostream> // Required for standard input/output operations.
#include <regex>    // Required for std::regex and std::regex_match.
#include <string>   // Required for std::string.

/**
 * @brief Main function to calculate the final value of variable X.
 * Functional Utility: Reads the number of operations and each operation string,
 *                     then uses regex matching to determine the operation type
 *                     and computes and prints the final value of X.
 */
int main() {
  int n;       // Stores the number of operations.
  int counter = 0; // Stores the current value of the variable X, initialized to 0.

  // Block Logic: Reads the total number of operations, `n`.
  std::cin >> n;

  /**
   * Block Logic: Iterates `n` times to process each operation.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `counter` correctly reflects the value of X after `i` operations.
   */
  for (int i = 0; i < n; i++) {
    std::string command; // Stores the operation string read from input.
    // Block Logic: Reads the current operation string.
    std::cin >> command;

    /**
     * Conditional Logic: Uses `std::regex_match` to determine if the `command` string
     *                    contains the "--" substring, which indicates a decrement operation.
     * `std::regex("(.*)(--)(.*)")` is a regex that matches any string containing "--".
     * If matched, it's a decrement; otherwise, it's an increment.
     */
    if (std::regex_match(command, std::regex("(.*)(--)(.*)")))
      counter--; // Decrement X.
    else
      counter++; // Increment X.
  }

  // Block Logic: Prints the final calculated value of X.
  std::cout << counter << std::endl;

  return 0; // Indicates successful program termination.
}
