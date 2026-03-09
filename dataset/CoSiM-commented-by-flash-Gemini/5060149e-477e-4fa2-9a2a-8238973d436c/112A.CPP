/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison after case normalization.
 *        This program reads two strings, converts them to lowercase,
 *        and then compares them lexicographically to determine their
 *        relative order.
 *
 * Algorithm: String input, case transformation using `std::transform` with `::tolower`,
 *            and direct lexicographical comparison using standard string operators.
 * Time Complexity: O(L) where L is the length of the longer string, due to
 *                  string transformation and comparison.
 * Space Complexity: O(L) for storing the input strings.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Initializes string variables, reads input,
   *                     normalizes case, and performs lexicographical comparison.
   */
  string input1, input2; // Variables to store the two input strings.

  // Block Logic: Reads two strings from standard input.
  // Pre-condition: Standard input contains two newline-separated strings.
  cin >> input1;
  cin >> input2;

  /**
   * Block Logic: Converts both input strings to lowercase for case-insensitive comparison.
   *              `std::transform` applies `::tolower` to each character in the string.
   * Pre-condition: `input1` and `input2` contain the original case strings.
   * Invariant: After transformation, `input1` and `input2` contain only lowercase characters.
   */
  transform(input1.begin(), input1.end(), input1.begin(), ::tolower);
  transform(input2.begin(), input2.end(), input2.begin(), ::tolower);

  /**
   * Block Logic: Compares the normalized strings lexicographically.
   *              Prints "0" if strings are equal, "1" if `input1` is lexicographically greater,
   *              and "-1" if `input2` is lexicographically greater.
   * Pre-condition: `input1` and `input2` are in lowercase.
   * Invariant: One of the three comparison outcomes is printed to standard output.
   */
  if (input1 == input2) {
    cout << "0" << endl; // Strings are identical after case normalization.
  } else if (input1 > input2) {
    cout << "1" << endl; // First string is lexicographically greater.
  } else {
    cout << "-1" << endl; // Second string is lexicographically greater.
  }

  return 0; // Indicates successful execution.
}
