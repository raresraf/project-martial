/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, shortening those that are "too long"
 *        according to a specific rule: words with length strictly greater than 10 are replaced
 *        by their first letter, the count of intermediate letters, and their last letter.
 *        Words with length 10 or less are printed without modification.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the number of words to process.
 *   2. For each word:
 *      a. Reads the word `s`.
 *      b. If its length is strictly greater than 10, prints the first character,
 *         then the numerical value of (length - 2), and finally the last character.
 *      c. If its length is 10 or less, prints the word as is.
 *
 * Time Complexity: O(L_total), where L_total is the sum of lengths of all input words,
 *                  as each character is processed at most a constant number of times.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word,
 *                   for storing the current word.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  /**
   * @brief Main function implementing the solution for the "Way Too Long Words" problem.
   *
   * Reads the number of words `n`, then processes each word according to the abbreviation rule.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n; // Total number of words to process.
  cin >> n; // Reads the number of words.

  /**
   * Block Logic: Iterates `n` times to process each word.
   * Invariant: In each iteration, one word is read and processed according to the rules.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) { // Loop continues for `n` iterations, decrementing `n` each time.
    string s; // Current word being processed.
    cin >> s; // Reads the current word.

    /**
     * Conditional Logic: Checks if the word's length exceeds 10 characters.
     * If true, it prints an abbreviated version; otherwise, it prints the original word.
     */
    if (s.length() > 10)
      // Prints the first character, then the number of intermediate characters (length - 2),
      // and finally the last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl; // Prints the original word if its length is 10 or less.
  }

  return 0;
}
