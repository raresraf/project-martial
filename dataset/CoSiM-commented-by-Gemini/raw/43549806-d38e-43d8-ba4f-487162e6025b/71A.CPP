/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *
 * @algorithm
 * The problem requires abbreviating words that have a length strictly greater
 * than 10. The abbreviation consists of the first character, the number of
 * characters between the first and last, and the last character. Shorter or
 * equal length words remain unchanged.
 *
 * The algorithm is as follows:
 * 1. Read the number of words to process, `t`.
 * 2. Loop `t` times.
 * 3. In each iteration, read the word into a character array.
 * 4. Calculate the word's length.
 * 5. If the length is greater than 10, print the abbreviation: the first
 *    character, the length minus 2, and the last character.
 * 6. Otherwise, print the word as is.
 *
 * @note This solution uses C-style strings (a `char` array) and `strlen`.
 *
 * @complexity
 * - Time Complexity: O(T * L_max), where T is the number of test cases (words)
 *   and L_max is the maximum length of a word. This is dominated by reading
 *   the input and `strlen`.
 * - Space Complexity: O(L_max) to store the longest word.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, len = 0;
  char word[101];
  cin >> t;
  /**
   * @brief This loop processes each of the `t` words.
   */
  while (t--) {
    cin >> word;
    len = strlen(word);
    /**
     * @brief Conditional logic for abbreviation.
     * If the word's length is 10 or less, print it as is.
     * Otherwise, print the abbreviated form.
     */
    if (len <= 10)
      cout << word << '\n';
    else
      // Abbreviation: first char, (length - 2), last char.
      cout << word[0] << len - 2 << word[len - 1] << '\n';
  }
}
