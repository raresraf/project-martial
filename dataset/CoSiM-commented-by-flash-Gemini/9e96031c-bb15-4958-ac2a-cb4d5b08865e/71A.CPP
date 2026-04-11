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
  // Functional Utility: 'w' temporarily stores each word read from input.
  string w;

  // Block Logic: Reads the number of words 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates 'n' times, processing one word in each iteration.
   * Invariant: At the start of each iteration, 'i' represents the number of words already processed.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads a word from standard input.
    cin >> w;
    /**
     * Block Logic: Checks if the length of the current word is greater than 10.
     * Pre-condition: 'w' contains the current word.
     * Invariant: Words with length > 10 are abbreviated; others are printed as is.
     */
    if (w.length() > 10) {
      // Functional Utility: Prints the first character, the count of intermediate characters, and the last character.
      // Inline: w.at(0) accesses the first character, w.length() - 2 calculates intermediate count, w.back() accesses the last character.
      cout << w.at(0) << w.length() - 2 << w.back() << endl;
    } else {
      // Functional Utility: Prints the word as is.
      cout << w << endl;
    }
  }
  return 0;
}
