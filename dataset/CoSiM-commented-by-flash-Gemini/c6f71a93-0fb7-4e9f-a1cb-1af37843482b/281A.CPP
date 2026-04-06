/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - "Word Capitalization".
 * This program capitalizes the first letter of a given input string, leaving
 * the rest of the characters unchanged. This is a common string manipulation
 * task in competitive programming.
 *
 * @problem_statement
 * Read a single word (string) and print it after capitalizing its first letter.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Access the first character of the string.
 * 3. Convert the first character to its uppercase equivalent using `toupper()`.
 * 4. Print the modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 *   Reading the string and printing it takes linear time. `toupper()` is O(1).
 * Space Complexity: O(L), to store the input string.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

/** @typedef vi
 * @brief Alias for `std::vector<int>`.
 * A common competitive programming shorthand to reduce verbosity when working
 * with vectors of integers.
 */
typedef vector<int> vi;

/** @def all(c)
 * @brief Macro for obtaining iterators to the beginning and end of a container.
 * Simplifies the syntax for passing ranges to algorithms like `std::sort`, `std::for_each`, etc.
 */
#define all(c) (c).begin(), (c).end()

/** @def pb
 * @brief Macro for `push_back`.
 * A common competitive programming shorthand for adding an element to the end
 * of a `std::vector`.
 */
#define pb push_back

/** @def tr(a)
 * @brief Debugging macro to print the contents of a vector.
 * Iterates through a given vector and prints each element, followed by a newline.
 * Useful for quickly inspecting intermediate states during competitive programming.
 * @param a The vector to be printed.
 */
#define tr(a)                                                                  \
  cout << "Vector: ";                                                          \
  for (auto i : a)                                                             \
    cout << i << " ";                                                          \
  cout << "\n";

int main() {
  /**
   * @brief Main function to capitalize the first letter of an input word.
   * Orchestrates input reading, character manipulation, and outputting the
   * modified string.
   * @return 0 on successful execution.
   */

  /**
   * @brief Optimizes C++ standard stream I/O for faster execution.
   * `ios_base::sync_with_stdio(false)` unties C++ streams from C standard I/O.
   * `cin.tie(NULL)` prevents `cout` from flushing before `cin` operations,
   * which is beneficial in competitive programming for performance.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string s; ///< @var s The input string (word) whose first letter is to be capitalized.
  cin >> s; // Read the input word.

  /**
   * @brief Capitalizes the first character of the input string.
   * @details The `toupper()` function converts a character to its uppercase equivalent.
   * This directly modifies the first character of the string `s`.
   */
  s[0] = toupper(s[0]);

  cout << s; // Print the modified string with the capitalized first letter.
  return 0;
}
