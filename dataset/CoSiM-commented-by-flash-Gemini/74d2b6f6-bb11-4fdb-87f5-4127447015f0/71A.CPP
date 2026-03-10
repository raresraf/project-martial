/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic, commonly found in competitive programming.
 *   If a word's length exceeds 10 characters, it is abbreviated to its first letter, followed by
 *   the number of characters between the first and last letters, and then its last letter.
 *   Otherwise, the word is printed as is.
 * * Algorithm: String length check and conditional abbreviation.
 * Time Complexity: O(T * L_max), where T is the number of test cases and L_max is the maximum string length.
 * Space Complexity: O(L_max) to store the longest input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to process a series of words and apply abbreviation rules.
 * Functional Utility: Handles multiple test cases, reading a string for each and applying the abbreviation logic.
 */
int main() {
  int t;
  string s;
  // Block Logic: Reads the number of test cases 't'.
  cin >> t;
  /**
   * Block Logic: Iterates through each test case.
   * Pre-condition: 't' is a non-negative integer representing the number of words to process.
   * Invariant: In each iteration, one word is read and processed according to its length.
   */
  while (t--) {
    // Block Logic: Reads the current string for processing.
    cin >> s;
    /**
     * Block Logic: Applies the abbreviation rule based on string length.
     * If the string length is greater than 10, it constructs an abbreviated string.
     * Otherwise, it prints the original string.
     */
    if (s.size() > 10) {
      cout << s[0] << s.size() - 2 << s.back() << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}