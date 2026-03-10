/**
 * @file 281A.CPP
 * @brief Solution for a competitive programming problem that capitalizes the first letter of a word.
 *
 * @algorithm Direct character manipulation by ASCII value.
 * @time_complexity O(1), as the operation is on a single character.
 * @space_complexity O(N), where N is the length of the input string.
 */
#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

/**
 * @block
 * @brief Redirects standard I/O to files for local testing.
 * This preprocessor directive is common in competitive programming to allow
 * for testing with file-based input and output when not running on the
 * online judge's system.
 */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s;
  cin >> s;
  int x;
  x = int(s[0]);

  /**
   * @block
   * @brief Capitalizes the first character of the string if it is a lowercase letter.
   * It checks the ASCII value of the first character to see if it falls within
   * the range of lowercase English letters (>= 97). If so, it converts it to
   * uppercase by subtracting 32 from its ASCII value.
   */
  if (x >= 97) {
    s[0] = x - 32;
  }
  cout << s;
}