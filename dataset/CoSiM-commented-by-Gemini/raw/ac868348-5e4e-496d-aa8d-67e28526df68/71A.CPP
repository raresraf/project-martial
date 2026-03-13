/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 *
 * This program reads a series of words and shortens any word that is longer
 * than 10 characters. The shortened word is represented by its first character,
 * the number of characters between the first and last character, and its last
 * character.
 *
 * @see https://codeforces.com/problemset/problem/71/A
 */
#include <iostream>
// #include <algorithm>
// #include <vector>
#include <set>
// #include <map>

// #define lli long long int

using namespace std;

int main() {
  int n, l;
  string s;
  cin >> n;
  /**
   * @brief Main processing loop.
   *
   * Iterates `n` times, reading a word in each iteration. It then checks the
   * length of the word and applies the shortening logic if necessary.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    l = s.length();
    /**
     * @brief String shortening logic.
     *
     * If the length of the string `s` is greater than 10, it prints a
     * shortened version. Otherwise, it prints the original string.
     */
    if (l > 10) {
      cout << s[0] << l - 2 << s[l - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
}
