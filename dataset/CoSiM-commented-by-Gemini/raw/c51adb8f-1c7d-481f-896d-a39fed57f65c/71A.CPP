/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" competitive programming problem.
 * @details This program processes a list of words. For any word longer than 10 characters,
 * it prints an abbreviation consisting of the first letter, the count of
 * letters in between, and the last letter. Words 10 characters or shorter are printed as-is.
 *
 * @section Algorithm
 * 1. Read the number of words, 'n'.
 * 2. Loop 'n' times to process each word:
 *    a. Read the word into a string.
 *    b. If the word's length is greater than 10, print the abbreviation
 *       (e.g., "localization" becomes "l10n").
 *    c. Otherwise, print the word unchanged.
 *
 * @section Complexity
 * - Time Complexity: O(N * L_avg), where N is the number of words and L_avg is their
 *   average length. The complexity is dominated by the string input operations.
 * - Space Complexity: O(L_max), where L_max is the maximum length of any single word
 *   read into memory.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // 'n' is the number of words to process.
  int n;
  cin >> n;

  string s;
  /**
   * Block Logic: Loop 'n' times to process each word.
   * Invariant: The loop executes for each of the 'n' words provided in the input.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    /**
     * Block Logic: Decide whether to abbreviate the word based on its length.
     * Pre-condition: 's' contains the word to be processed.
     * The condition checks if the word length exceeds the threshold of 10.
     */
    if (s.size() <= 10) {
      // Post-condition: If the word is 10 characters or less, print it directly.
      cout << s << endl;
    } else {
      // Post-condition: If the word is longer than 10 characters, print the abbreviation.
      // The abbreviation consists of the first character, the number of internal characters (length - 2),
      // and the last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }
}
