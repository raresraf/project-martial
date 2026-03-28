/**
 * @file 282A.CPP
 * @brief An incorrect attempt to solve the "Bit++" programming problem.
 *
 * The program is intended to simulate a series of increment and decrement
 * operations on a variable `x`, which starts at 0. The operations are given
 * as strings like "++X", "X++", "--X", and "X--".
 *
 * @b WARNING: This program contains a critical bug. It increments the variable `x`
 *    for both increment ("++") and decrement ("--") operations, leading to an
 *    incorrect final result.
 *
 * @b Intended_Algorithm: The goal is to read a number of operations and apply them
 *    to a counter, then print the final value.
 * @b Time_Complexity: O(T), where T is the number of operations, due to the main loop.
 * @b Space_Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int x = 0; // Initialize the variable.
  int t;     // Number of operations.
  cin >> t;
  
  while (t)
  {
    string str;
    cin >> str;

    // Correctly handles the increment operations.
    if (str == "X++" || str == "++X") {
      x = x + 1;
    }

    // BUG: This block should decrement `x` but instead increments it.
    // The line should be `x = x - 1;` or `x--;`.
    if (str == "X--" || str == "--X") {
      x = x + 1; // This is incorrect.
    }
    t--;
  }
  
  // Prints the final, incorrect value of x.
  cout << x;
  return 0;
}