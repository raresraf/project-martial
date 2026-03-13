/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces.
 *
 * This program reads a single word and capitalizes its first letter, leaving
 * the rest of the word unchanged.
 *
 * @see https://codeforces.com/problemset/problem/281/A
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;

  cin >> s;

  /**
   * @brief Capitalizes the first letter of the string.
   *
   * The `toupper` function is used to convert the character at the first
   * position of the string to its uppercase equivalent.
   */
  s[0] = toupper(s[0]);

  cout << s << endl;
}
