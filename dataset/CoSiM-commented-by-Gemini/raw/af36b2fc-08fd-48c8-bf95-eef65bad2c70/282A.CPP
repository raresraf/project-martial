/**
 * @file 282A.CPP
 * @brief Solves the Bit++ problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/282/A
 *
 * This program calculates the final value of a variable 'x' after a series
 * of increment and decrement operations. The operations are given as strings
 * like "++X", "X++", "--X", or "X--".
 */
// https://codeforces.com/problemset/problem/282/A
// 282A  Bit++
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string str;
  // i will be used for increments and j for decrements.
  // A single variable would be clearer here.
  int i = 0, j = 0;
  while (n--) {
    cin >> str;
    /**
     * @brief The core logic relies on the fact that increment operations
     * ("++X" and "X++") have a '+' at the second position, while
     * decrement operations have a '-'.
     */
    if (str[1] == '+') {
      i++;
    } else
      j--;
  }
  /**
   * @brief The final value is the sum of increments and decrements.
   * Since j is decremented from 0, it holds the total number of decrements as a negative value.
   */
  cout << i + j;
}
