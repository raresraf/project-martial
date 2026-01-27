/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details This program compares two strings lexicographically, ignoring case.
 * It converts both strings to lowercase and then compares them.
 * The output is -1 if the first string is less than the second, 1 if it's greater, and 0 if they are equal.
 *
 * @see https://codeforces.com/problemset/problem/112/A
 *
 * Algorithm:
 * 1. Read two strings, `a` and `b`.
 * 2. Convert both strings to lowercase to ensure a case-insensitive comparison.
 * 3. Compare the modified strings.
 * 4. If `a` is lexicographically smaller than `b`, print -1.
 * 5. If `a` is lexicographically greater than `b`, print 1.
 * 6. If they are equal, print 0.
 *
 * Time Complexity: O(L), where L is the length of the strings. This is because the program iterates through the strings once for conversion and once for comparison.
 * Space Complexity: O(L) to store the two strings.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;

  /**
   * Block Logic: Convert both strings to lowercase.
   * This loop iterates through each character of the strings and converts it to its lowercase equivalent
   * to ensure the comparison is case-insensitive.
   */
  for (int i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }

  /**
   * Block Logic: Compare the two strings lexicographically.
   * If the strings are identical, the result is 0.
   * Otherwise, the program iterates through the strings to find the first differing character
   * and determines which string is lexicographically smaller or larger.
   */
  if (a == b)
    cout << "0" << endl;
  else {
    for (int i = 0; i < a.size(); i++) {
      if (a[i] < b[i]) {
        cout << "-1" << endl;
        break;
      }
      if (a[i] > b[i]) {
        cout << "1" << endl;
        break;
      }
    }
  }
  return 0;
}