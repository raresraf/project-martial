/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, shortening those that are "too long"
 *        according to a specific rule: words with length strictly greater than 10 are replaced
 *        by their first letter, the count of intermediate letters, and their last letter.
 *        Words with length 10 or less are printed without modification.
 * Algorithm:
 *   1. Read an integer 'n', representing the number of words to process.
 *   2. Loop 'n' times:
 *      a. Read a word (string `str`).
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
   * Reads a number of words (n) and applies an abbreviation rule to each:
   * words with length greater than 10 characters are abbreviated to their first
   * letter, the count of intermediate characters, and their last letter.
   * Words with length 10 or less are printed without modification.
   * @return int Returns 0 upon successful execution.
   */
  int n;
  cin >> n; // Reads the number of words to be processed.

  string str; // Declare string to store each word.

  // Block Logic: Iterates 'n' times to process each input word.
  // Pre-condition: 'n' specifies the total number of words.
  // Invariant: For each iteration, a word is read and either printed directly or abbreviated.
  for (int i = 0; i < n; i++) {
    cin >> str; // Reads the current word.

    // Block Logic: Checks if the word's length is greater than 10 to decide on abbreviation.
    // If true, abbreviates the word; otherwise, prints it as is.
    if (str.size() <= 10) { // If the word is short enough.
      cout << str << endl; // Prints the word without modification.
    } else {
      // Block Logic: Abbreviates the word to its first character, (length-2) as a number, and its last character.
      // Example: "localization" (length 12) becomes "l10n".
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;
    }
  }

  return 0;
}
