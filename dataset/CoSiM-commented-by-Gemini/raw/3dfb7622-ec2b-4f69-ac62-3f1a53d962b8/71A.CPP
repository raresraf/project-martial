/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (Codeforces 71A).
 *
 * @algorithm
 * The problem requires shortening words that are "too long". A word is too long
 * if its length is strictly greater than 10 characters.
 * The abbreviation for a long word is formed by its first character, followed
 * by the number of letters between the first and last character, and finally its
 * last character. For example, "localization" becomes "l10n". Words with 10 or
 * fewer characters remain unchanged.
 *
 * The algorithm is as follows:
 * 1. Read the number of words to process, `t`.
 * 2. Loop `t` times.
 * 3. In each iteration, read the input word.
 * 4. Check if the word's length is greater than 10.
 * 5. If it is, construct and print the abbreviation.
 * 6. If not, print the word as is.
 *
 * @complexity
 * - Time Complexity: O(T * L), where T is the number of words and L is the
 *   maximum length of a word, due to reading the input strings. The processing
 *   for each word is effectively O(1) after it's read.
 * - Space Complexity: O(L) to store the longest input word.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int t; // The number of words to process.
  cin >> t;
  /**
   * @brief Main loop to process each of the `t` words.
   */
  for (int i = 0; i < t; i++) {
    string s;
    cin >> s;
    /**
     * @brief Check if the word is longer than 10 characters.
     */
    if (s.length() > 10)
      /**
       * @brief If the word is too long, print its abbreviation:
       * - The first character: s[0]
       * - The number of characters between the first and last: s.length() - 2
       * - The last character: s[s.length() - 1]
       */
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      // If the word is not too long, print it unchanged.
      cout << s << endl;
  }
}
