/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 *
 * This program reads a series of words and shortens any word that is longer
 * than 10 characters. The abbreviated form consists of the first character,
 * the number of characters between the first and last, and the last character.
 *
 * Algorithm: String manipulation based on length.
 * Time Complexity: O(N * L) where N is the number of words. Given that the
 *                  length of the number of middle characters is small, this
 *                  approaches O(N).
 * Space Complexity: O(L_max) where L_max is the length of the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n; // The number of words to process.
  cin >> n;

  string s; // Variable to store each word.
  
  /**
   * @brief Loop to process each of the 'n' words.
   *
   * Pre-condition: 'n' contains the total count of words.
   * Invariant: On each iteration, one word is read, processed, and printed.
   */
  for (int i = 0; i < n; ++i) {
    cin >> s;

    /**
     * @brief Conditional block to check word length.
     * Logic: If the word's length exceeds 10, it is abbreviated.
     *        Otherwise, it is printed as is.
     */
    if (s.size() > 10) {
      // Print the abbreviated form: first char, count of middle chars, last char.
      cout << s[0] << s.size() - 2 << s.back();
    } else {
      cout << s;
    }

    cout << '
';
  }
}
