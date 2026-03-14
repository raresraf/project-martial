/**
 * @file 282A.CPP
 * @brief Solves the Bit++ problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/282/A
 *
 * This program calculates the final value of a variable 'x' after a series
 * of increment and decrement operations. The operations are given as strings
 * like "++X", "X++", "--X", or "X--".
 */
#include <iostream>
using namespace std;

int main() {

  int n;
  int ans = 0;
  string str;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    /**
     * @brief Checks the input string to determine whether to increment or decrement.
     * This version explicitly checks for the increment operations.
     */
    if (str == "X++" || str == "++X")
      ans++;
    else
      ans--;
  }

  cout << ans << endl;
  return 0;
}
