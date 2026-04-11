/**
 * @file 71A.CPP
 * @brief This program processes a list of words. For each word, if its length
 * is greater than 10, it abbreviates the word to its first character, the
 * count of intermediate characters, and its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 *
 * Algorithm: String length check and conditional string manipulation.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) to store the longest word.
 */

#include <bits/stdc++.h> // Includes iostream, string, etc.

using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of words to be processed.
  int n;
  // Block Logic: Reads the number of words 'n' from standard input.
  cin >> n;

  // Functional Utility: 's' temporarily stores each word read from input.
  string s;
  /**
   * Block Logic: Iterates 'n' times, processing one word in each iteration.
   * Invariant: At the start of each iteration, 'i' represents the number of words already processed.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads a word from standard input.
    cin >> s;
    /**
     * Block Logic: Checks if the length of the current word is less than or equal to 10.
     * Pre-condition: 's' contains the current word.
     * Invariant: Words with length <= 10 are printed as is.
     */
    if (s.size() <= 10) {
      // Functional Utility: Prints the word as is.
      cout << s << endl;
    }
    /**
     * Block Logic: If the word's length is greater than 10, it's abbreviated.
     * Pre-condition: 's' has a length greater than 10.
     * Invariant: The abbreviated form (first_char + (length-2) + last_char) is printed.
     */
    else {
      // Functional Utility: Prints the first character, the count of intermediate characters, and the last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }
}
