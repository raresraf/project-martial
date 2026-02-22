/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *
 * @algorithm
 * The problem is to abbreviate words that are longer than 10 characters.
 * The abbreviation is formed by taking the first character, followed by the
 * count of intermediate characters (i.e., length - 2), and finally the last
 * character. Words with a length of 10 or less are not changed.
 *
 * The algorithm is as follows:
 * 1. Read the number of words, `t`.
 * 2. Loop `t` times to process each word.
 * 3. Inside the loop, read the word as a `std::string`.
 * 4. Get the length of the string.
 * 5. If the length is greater than 10, print the abbreviation.
 * 6. Otherwise, print the original string.
 *
 * @complexity
 * - Time Complexity: O(T * L_max), where T is the number of words and L_max
 *   is the maximum length of a word, dominated by input/output operations.
 * - Space Complexity: O(L_max) to store the longest word string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  /**
   * @brief Loop to process each of the `t` test cases (words).
   */
  while (t--) {
    string s;
    cin >> s;
    int n = s.length();
    /**
     * @brief Check if the word is "too long" (length > 10).
     */
    if (n <= 10) {
      // If the word is not too long, print it as is.
      cout << s << endl;
    } else {
      // If the word is too long, print its abbreviation.
      // Format: first character, count of middle characters, last character.
      cout << s[0] << n - 2 << s[n - 1] << endl;
    }
  }
  return 0;
}
