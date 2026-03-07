/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 *
 * @details
 * This program shortens words that are longer than 10 characters by replacing
 * the middle characters with a count of their number. Shorter words are
 * printed unchanged.
 *
 * Algorithm:
 * 1. Read the number of words to process, `t`.
 * 2. Loop `t` times:
 *    a. Read a word.
 *    b. If the word's length is 10 or less, print the word as is.
 *    c. Otherwise, print the first character, followed by the number of
 *       middle characters (length - 2), followed by the last character.
 *
 * Time Complexity: O(T * L), where T is the number of words and L is the
 *                  maximum word length, dominated by I/O operations.
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  // Loop 't' times to process each word.
  while (t--) {
    string s;
    cin >> s;
    // Check if the word is "too long" (length > 10).
    if (s.length() < 11) // Equivalent to s.length() <= 10
      cout << s << endl;
    else
      // Print the abbreviation: first char, length-2, last char.
      // s.back() is a C++11 feature to get the last character.
      cout << s[0] << s.length() - 2 << s.back() << endl;
  }

  return 0;
}