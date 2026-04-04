/**
 * @file 71A.CPP
 * @brief Implements "Way Too Long Words" problem from Codeforces with I/O optimization.
 *
 * This program reads an integer 'n' representing the number of test cases.
 * For each test case, it reads a string 's'. If the length of 's' is strictly
 * greater than 10, the program prints an abbreviated version of the word:
 * the first character, followed by the number of characters between the first
 * and last (length - 2), followed by the last character. Otherwise, it prints
 * the word as is. This version uses standard competitive programming I/O optimizations.
 *
 * Algorithm: String length check and conditional formatting.
 * Time Complexity: O(L) for each test case, where L is the length of the string,
 *                  dominated by string input/output operations. Total O(N * L_avg).
 * Space Complexity: O(L_max) for storing the longest string.
 */
#include <bits/stdc++.h> // Includes most standard library headers.
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Processes multiple test cases, abbreviating long words.
   * @return 0 implied by function end.
   */
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  // 'ios::sync_with_stdio(false)' unties C++ streams from C standard streams.
  // 'cin.tie(0)' prevents 'cin' from flushing 'cout' before each input operation.
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n;     // Variable to store the number of test cases.
  cin >> n;  // Read the number of test cases.

  /**
   * @brief Loop to process each test case.
   * @pre 'n' is a non-negative integer.
   * @post For each test case, an abbreviated or original string is printed.
   * @invariant 'n' decrements in each iteration until all test cases are processed.
   */
  while (n--) {
    string s;   // Variable to store the input string for each test case.
    cin >> s;       // Read the input string for the current test case.

    int t = s.length(); // Get the length of the string.

    /**
     * @brief Conditional block to check string length and print accordingly.
     * @pre 't' holds the length of the string 's'.
     */
    if (t > 10) // If the word is too long (length greater than 10).
      // Functional Utility: Prints the abbreviated word format: first_char + (length-2) + last_char.
      // 's.at(index)' provides bounds checking, similar to 's[index]' but throws an exception if out of bounds.
      cout << s.at(0) << t - 2 << s.at(t - 1) << "\n";
    else // If the word is not too long (length 10 or less).
      cout << s << "\n"; // Print the original word.
  }
}
