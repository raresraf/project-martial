/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program processes a list of words and shortens any word that
 * has a length strictly greater than 10.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of words to process.
 * 2. Loop `n` times.
 * 3. In each iteration, read a word into a string `s`.
 * 4. Check if the length of `s` is greater than 10 (or `< 11`).
 * 5. If the length is 10 or less, print the string as is.
 * 6. If the length is greater than 10, construct and print an abbreviated
 *    version. The abbreviation consists of the first character, followed by
 *    the number of characters between the first and last (i.e., length - 2),
 *    followed by the last character.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is their
 * maximum length, due to string I/O and processing.
 * Space Complexity: O(L) to store each word.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  /**
   * @brief This loop reads and processes each of the `n` words.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    /**
     * @brief This block checks the word's length and applies the problem's
     * rule: print the word if its length is 10 or less, otherwise
     * print its abbreviation.
     */
    if (s.length() < 11)
      cout << s << endl;
    else {
      // Construct the abbreviation: first char, count of middle chars, last char.
      string r = s[0] + to_string(s.length() - 2) + s[s.length() - 1];
      cout << r << endl;
    }
  }
}