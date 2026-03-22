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
 *      a. Reads the word.
 *      b. If its length is strictly greater than 10, prints the first character,
 *         then the numerical value of (length - 2), and finally the last character.
 *      c. If its length is 10 or less, prints the word as is.
 *
 * Time Complexity: O(L_total), where L_total is the sum of lengths of all input words,
 *                  as each character is processed at most a constant number of times.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word,
 *                   for storing the current word.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function implementing the solution for the "Way Too Long Words" problem.
   *
   * Sets up input reading, then processes `n` words according to the abbreviation rule.
   * Uses `scanf` for integer input and `cin` for string input.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n;
  string s;
  scanf("%d", &n); // Reads the number of words to process.
  /**
   * Block Logic: Iterates `n` times to process each word.
   * Invariant: In each iteration, one word is read and processed.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) {
    cin >> s; // Reads the current word.
    /**
     * Conditional Logic: Checks if the word's length exceeds 10 characters.
     * Pre-condition: `s` contains the current word.
     * Post-condition: Abbreviated word or original word is printed.
     */
    if (s.length() > 10)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl;
  }
}
