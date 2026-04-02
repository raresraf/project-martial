/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details Rearranges a string representing a sum of numbers (1, 2, 3) into non-decreasing order.
 *
 * Algorithm: Counting Sort
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(1) auxiliary space.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Optimizes I/O operations by decoupling C++ streams from C standard streams.
   */
  ios_base::sync_with_stdio(0);
  string s;
  cin >> s;
  int a = 0, b = 0, c = 0;
  /**
   * @brief Counts the occurrences of the numbers '1', '2', and '3' in the input string.
   * This is the core of the counting sort algorithm.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      a++;
    if (s[i] == '2')
      b++;
    if (s[i] == '3')
      c++;
  }
  /**
   * @brief Reconstructs the string by appending the numbers in sorted order, separated by '+'.
   * It first appends all '1's, then '2's, then '3's.
   */
  for (int i = 0; i < a; i++) {
    if (i == a - 1)
      cout << "1";
    else
      cout << "1+";
  }
  if (b != 0) {
    if (a != 0)
      cout << "+";
    for (int i = 0; i < b; i++) {
      if (i == b - 1)
        cout << "2";
      else
        cout << "2+";
    }
  }
  if (c != 0) {
    if (a != 0 || b != 0)
      cout << "+";
    for (int i = 0; i < c; i++) {
      if (i == c - 1)
        cout << "3";
      else
        cout << "3+";
    }
  }

  return 0;
}
