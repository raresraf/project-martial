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

#include <iostream>
#include <string> // Required for std::string
using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of words to be processed.
  int n;
  // Functional Utility: 'i' is a loop counter for iterating through the words.
  int i;
  // Functional Utility: 'l' is declared but unused in this code.
  int l; // Unused variable
  // Functional Utility: 's' temporarily stores each word read from input.
  string s;

  // Block Logic: Reads the number of words 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates 'n' times, processing one word in each iteration.
   * Invariant: At the start of each iteration, 'i' represents the number of words already processed.
   */
  for (i = 0; i < n; i++) {
    // Block Logic: Reads a word from standard input.
    cin >> s;
    /**
     * Block Logic: Checks if the length of the current word is greater than 10.
     * Pre-condition: 's' contains the current word.
     * Invariant: Words with length > 10 are abbreviated; others are printed as is.
     */
    if (s.length() > 10) {
      // Functional Utility: Prints the first character, the count of intermediate characters, and the last character.
      // Inline: s[0] accesses the first character, s.length() - 2 calculates intermediate count, s[s.length() - 1] accesses the last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else
      // Functional Utility: Prints the word as is.
      cout << s << endl;
  }

  return 0;
}
