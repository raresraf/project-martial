/**
 * @file 71A.CPP
 * @brief This program processes a list of words. For each word, if its length
 * is greater than 10, it abbreviates the word to its first character, the
 * count of intermediate characters, and its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is. The word processing logic is encapsulated
 * in a separate function.
 *
 * Algorithm: String length check and conditional string manipulation.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) to store the longest word during processing.
 */

#include <bits/stdc++.h> // Includes iostream, string, etc.
using namespace std;

/**
 * @brief Processes a single word: abbreviates if longer than 10 characters, otherwise prints as is.
 *
 * Pre-condition: Standard input contains a string (word).
 * Post-condition: The (potentially abbreviated) word is printed to standard output.
 */
void change() {
  // Functional Utility: 's' stores the input word.
  string s;
  // Block Logic: Reads a word from standard input.
  cin >> s;
  // Functional Utility: 'size' stores the length of the input word.
  int size = s.size();
  /**
   * Block Logic: Checks if the length of the word is greater than 10.
   * Invariant: Words with length > 10 are abbreviated; others are printed as is.
   */
  if (size > 10) {
    // Functional Utility: Prints the first character, the count of intermediate characters, and the last character.
    // Inline: s[0] accesses the first character, size - 2 calculates intermediate count, s[size - 1] accesses the last character.
    cout << s[0] << size - 2 << s[size - 1] << endl;
  } else {
    // Functional Utility: Prints the word as is.
    cout << s << endl;
  }
}

int main() {
  // Functional Utility: 'n' stores the total number of words to be processed.
  int n;
  // Block Logic: Reads the number of words 'n' from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, calling the 'change' function for each word.
   * Invariant: After each call to 'change()', one word has been processed and printed.
   */
  while (n--) {
    change();
  }
}
