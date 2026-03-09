/**
 * @file 282A.CPP
 * @brief Solution to Codeforces Problem 282A: "Bit++".
 *        This program simulates a simple programming language with a single variable `X`.
 *        Operations are either "++X", "X++", "--X", or "X--". The program reads a number of
 *        statements and calculates the final value of `X` starting from 0.
 *
 * Algorithm: Iterative processing of string statements. It checks for the presence of "++"
 *            to determine if `X` should be incremented, otherwise it's decremented.
 * Time Complexity: O(N * L) where N is the number of statements and L is the maximum length of a statement string.
 *                  Due to `string::find` operation.
 * Space Complexity: O(L_max) to store the longest statement string.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the main logic flow to read statements,
   *                     process them, and output the final value of X.
   */
  ios_base::sync_with_stdio(false); // Unties C++ standard streams from C standard streams for faster I/O.
  cin.tie(NULL); // Unties cin from cout, preventing flushing of cout before each cin operation.
  string in;
  int n;
  cin >> n;
  int res = 0;
  while (n--) {
    cin >> in;
    if (in.find("++") != in.npos)
      res++;
    else
      res--;
  }
  cout << res;
}
