/**
 * @file 282A.CPP
 * @brief Calculates the final value of a variable after a series of operations.
 *
 * This program simulates a series of increment (++) and decrement (--)
 * operations on a variable, which starts at 0. It reads the number of
 * operations and then the operations themselves, finally printing the
 * resulting value.
 *
 * @see https://codeforces.com/problemset/problem/282/A
 */
#include <iostream>
using namespace std;
int main() {
  int t;
  cin >> t;
  string x;
  int p = 0, m = 0;
  /**
   * @brief Main processing loop.
   *
   * Reads each operation as a string. It checks the second character of the
   * string to determine if it's an increment ('+') or decrement ('-')
   * operation and updates the respective counters.
   */
  while (t > 0) {
    cin >> x;
    if (x[1] == '+') {
      p++;
    } else {
      m++;
    }
    t--;
  }
  // The final value is the number of increments minus the number of decrements.
  cout << p - m;
  return 0;
}
