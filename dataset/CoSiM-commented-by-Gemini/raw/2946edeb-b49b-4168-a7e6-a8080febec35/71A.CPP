/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (71A) from Codeforces.
 * @details This program abbreviates words that are longer than 10 characters by
 * showing the first and last letter and the count of letters in between.
 *
 * @author @Nishant (as per original comment)
 *
 * @algorithm The solution iterates through a series of words and applies an
 * abbreviation rule based on length.
 * 1. Read the number of test cases, `t`.
 * 2. Loop `t` times to process each word.
 * 3. Read the word `s`.
 * 4. If the length of `s` is greater than 10:
 *    a. Print the abbreviation by concatenating the first character, the number
 *       of middle characters (`length - 2`), and the last character.
 * 5. Otherwise, print the original word `s`.
 *
 * @complexity
 * Time Complexity: O(T * L), where T is the number of words and L is the maximum length
 * of a word. The complexity is determined by the I/O operations.
 * Space Complexity: O(L), to store the longest input word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  /**
   * @block
   * @brief Processes 't' words, one per iteration.
   */
  while (t--) {
    string s;
    cin >> s;
    /**
     * @block
     * @brief Abbreviates the word if its length is > 10, otherwise prints it as is.
     */
    if (s.size() > 10) {
      // Construct the abbreviation by printing its parts directly.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}