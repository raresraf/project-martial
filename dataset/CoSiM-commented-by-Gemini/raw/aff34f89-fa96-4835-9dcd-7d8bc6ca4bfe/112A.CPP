/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/112/A
 *
 * This program performs a case-insensitive lexicographical comparison of two strings.
 */
// By archit-dev, contest: Codeforces Beta Round #85 (Div. 2 Only), problem: (A)
// Petya and Strings, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  // flag will store the result of the comparison:
  // 0 for equal, 1 for s1 > s2, -1 for s1 < s2.
  int flag = 0;
  string s1, s2;
  cin >> s1 >> s2;
  
  // Convert both strings to uppercase for case-insensitive comparison.
  for (int i = 0; i < s1.size(); i++) {
    s1[i] = toupper(s1[i]);
  }
  for (int i = 0; i < s2.size(); i++) {
    s2[i] = toupper(s2[i]);
  }
  
  /**
   * @brief Lexicographically compare the two strings character by character.
   */
  for (int i = 0; i < s1.size(); i++) {
    if (s1[i] > s2[i]) {
      flag = 1;
      break;
    } else if (s1[i] < s2[i]) {
      flag = -1;
      break;
    } else {
      continue;
    }
  }
  cout << flag << "
";
  return 0;
}
