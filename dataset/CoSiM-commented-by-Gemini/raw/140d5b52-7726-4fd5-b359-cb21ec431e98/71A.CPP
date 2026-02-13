/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 * @details This program processes a list of words. If a word's length is greater than 10 characters,
 * it is abbreviated. The abbreviation consists of the first character, the number of characters
 * between the first and last (i.e., length - 2), and the last character. Otherwise, the word is printed as is.
 *
 * Algorithm:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times:
 *    a. Read the word into a string `s`.
 *    b. If the length of `s` is greater than 10:
 *       - Print the first character of `s`.
 *       - Print the length of `s` minus 2.
 *       - Print the last character of `s`.
 *    c. Else (if the length is 10 or less):
 *       - Print the string `s` as is.
 *
 * Time Complexity: O(N * L_avg), where N is the number of words and L_avg is the average length of a word, dominated by input operations.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word, used to store the word.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n; // Read the number of words to process.

  /**
   * @brief Main loop to process each word.
   * Iterates `n` times, once for each word.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s; // Read the current word.

    /**
     * @brief Check if the word is "too long" and needs to be abbreviated.
     * A word is considered too long if its length is strictly greater than 10.
     */
    if (s.length() > 10) {
      // If the word is too long, abbreviate it.
      // The format is the first letter, the count of letters in between, and the last letter.
      cout << s[0];
      cout << s.length() - 2;
      cout << s[s.length() - 1] << "\n";
    } else {
      // If the word is not too long, print it without any changes.
      cout << s << "\n";
    }
  }
}
