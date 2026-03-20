/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, shortening those that are "too long"
 *        according to a specific rule: words with length greater than 10 are replaced
 *        by their first letter, the count of intermediate letters, and their last letter.
 * Algorithm:
 *   1. Read an integer 'n', representing the number of words to process.
 *   2. Loop 'n' times:
 *      a. Read a word (string 's').
 *      b. Get the length of the word.
 *      c. If the length is less than or equal to 10, print the word as is.
 *      d. If the length is greater than 10, print the first character, then the integer
 *         (length - 2), and finally the last character.
 * Time Complexity: O(L_total), where L_total is the sum of lengths of all input words,
 *                  as each character is processed at most a constant number of times.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word,
 *                   for storing the current word.
 */
#include <iostream>
using namespace std;
int main() {
  /**
   * @brief Main function implementing the solution for the "Way Too Long Words" problem.
   *
   * Reads a number of words and applies a specific abbreviation rule:
   * words longer than 10 characters are abbreviated to their first letter,
   * the count of intermediate characters, and their last letter.
   * Shorter words are printed without modification.
   * @return int Returns 0 upon successful execution.
   */
  string s;
  int n;
  cin >> n;
  int len = 0;
  // Block Logic: Iterates 'n' times to process each word.
  // Pre-condition: 'n' specifies the total number of words to process.
  // Invariant: 'i' tracks the current word being processed from 0 to n-1.
  for (int i = 0; i < n; i++) {
    cin >> s;
    len = s.length();
    // Block Logic: Checks if the word's length exceeds the abbreviation threshold.
    // If true, abbreviates the word; otherwise, prints it as is.
    if (len <= 10)
      cout << s << endl;
    else {
      // Block Logic: Abbreviates the word to its first character, length-2, and last character.
      // E.g., "localization" becomes "l10n".
      cout << s[0] << len - 2 << s[len - 1] << endl;
    }
  }

  return 0;
}
