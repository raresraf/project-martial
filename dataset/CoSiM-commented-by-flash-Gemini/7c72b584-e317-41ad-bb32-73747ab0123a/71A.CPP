/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, shortening those that are "too long"
 *        according to a specific rule: words with length strictly greater than 10 are replaced
 *        by their first letter, the count of intermediate letters, and their last letter.
 *        Words with length 10 or less are printed without modification. This version
 *        uses string concatenation for building the abbreviated word.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the number of words to process.
 *   2. For each word:
 *      a. Reads the word `s`.
 *      b. If the length of `s` is less than or equal to 10, prints the word as is.
 *      c. If the length of `s` is strictly greater than 10, constructs a new string `r` by
 *         concatenating `s[0]`, the string representation of `s.length() - 2`, and `s[s.length() - 1]`.
 *         Then, prints `r`.
 *
 * Time Complexity: O(L_total), where L_total is the sum of lengths of all input words.
 *                  String operations like `to_string` and concatenation have complexity proportional
 *                  to the length of the string being created, but overall processing per character
 *                  remains constant.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word,
 *                   for storing the current word `s` and the potentially abbreviated result `r`.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function implementing the solution for the "Way Too Long Words" problem.
   *
   * Reads the number of words `n`, then processes each word to either print it as is
   * or a shortened version based on its length.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n;
  cin >> n;
  /**
   * Block Logic: Iterates `n` times to process each word.
   * Invariant: In each iteration, one word is read and processed according to the rules.
   * Pre-condition: `n` is a non-negative integer.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s; // Reads the current word.
    /**
     * Conditional Logic: Checks if the word's length is less than or equal to 10.
     * Pre-condition: `s` contains the current word.
     * Post-condition: Abbreviated word or original word is printed.
     */
    if (s.length() < 11) // Equivalent to s.length() <= 10
      cout << s << endl;
    else {
      // String Construction: Builds the abbreviated word using string concatenation.
      // s[0] is the first character.
      // to_string(s.length() - 2) converts the count of middle characters to a string.
      // s[s.length() - 1] is the last character.
      string r = s[0] + to_string(s.length() - 2) + s[s.length() - 1];
      cout << r << endl;
    }
  }
}