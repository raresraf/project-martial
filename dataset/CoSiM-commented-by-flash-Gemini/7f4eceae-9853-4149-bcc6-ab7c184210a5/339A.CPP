/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *        This program takes a mathematical sum consisting of numbers 1, 2, and 3
 *        separated by '+' signs, sorts the numbers in non-decreasing order,
 *        and then prints the sorted sum. This particular implementation uses a
 *        bubble sort-like approach to sort the digits in place.
 *
 * Algorithm: In-place Sorting (Bubble Sort variant)
 *   1. Reads an input string `s`, which contains digits ('1', '2', '3') and '+' signs.
 *   2. Uses nested loops to iterate through all pairs of characters in the string.
 *   3. For each pair `(s[i], s[j])`, it checks if both `s[i]` and `s[j]` are digits
 *      (i.e., not '+') and if `s[j]` is lexicographically smaller than `s[i]`.
 *   4. If these conditions are met, it swaps `s[i]` and `s[j]`. This effectively
 *      sorts the digits while preserving the positions of '+' signs.
 *   5. After the sorting process, prints the modified string `s`.
 *
 * Time Complexity: O(L^2), where L is the length of the input string `s`, due to the nested loops.
 * Space Complexity: O(L) for storing the input string `s`. O(1) for auxiliary variables.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Helpful Maths" problem.
   *
   * Reads a sum string, sorts the numeric digits in non-decreasing order
   * using an in-place comparison sort, and prints the sorted sum.
   *
   * @return int Returns 0 upon successful execution.
   */
  int i, j; // Loop counters.
  char temp; // Temporary variable for swapping characters.
  string s;  // Input string (e.g., "3+2+1").
  cin >> s;  // Reads the input string.
  /**
   * Block Logic: Implements a bubble sort-like algorithm to sort the digits in the string.
   * It iterates through all possible pairs of digits, skipping '+' characters.
   * Invariant: After each outer loop iteration, the largest unsorted digit bubbles
   *            to its correct position.
   */
  for (i = 0; i < s.size(); i++) {
    for (j = i + 1; j < s.size(); j++) {
      /**
       * Conditional Logic: Checks if both characters are digits and if they are out of order.
       * `s[i] != '+' && s[j] != '+'` ensures only digits are compared.
       * `s[j] < s[i]` checks for the sorting order (non-decreasing).
       */
      if (s[i] != '+' && s[j] != '+' && s[j] < s[i]) {
        temp = s[j]; // Store `s[j]`.
        s[j] = s[i]; // Replace `s[j]` with `s[i]`.
        s[i] = temp; // Replace `s[i]` with the stored `s[j]`.
      }
    }
  }
  // Prints the modified string with digits sorted.
  cout << s;

  return 0;
}
