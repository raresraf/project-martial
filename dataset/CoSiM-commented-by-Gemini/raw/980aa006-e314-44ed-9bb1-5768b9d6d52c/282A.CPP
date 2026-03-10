/**
 * @file 282A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Bit++").
 * This program simulates a series of increment and decrement operations on a
 * single variable `x`, which starts at 0, and prints its final value.
 *
 * @algorithm Simulation. It reads each operation as a string and updates a
 * counter based on whether the operation is an increment or a decrement.
 *
 * @time_complexity O(N), where N is the number of operations, as each operation is processed in constant time.
 * @space_complexity O(1) for the counter, plus storage for one operation string at a time.
 */
/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/282/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Optimizes C++ stream I/O for speed.
  ios::sync_with_stdio(0);
  cin.tie(0);

  int16_t x{0}; // The variable to be manipulated, initialized to 0.
  string e;     // String to hold the current operation.

  // The first line of input (the number of operations, n) is ignored.
  // The loop will read `n` times anyway until input ends.
  cin.ignore();

  /**
   * @block
   * @brief Reads and executes each operation.
   * This loop continues as long as there is input. It reads each statement
   * (e.g., "X++", "--X") and modifies `x` accordingly.
   */
  while (cin >> e) {
    if (e == "X++" || e == "++X")
      x++;
    else if (e == "X--" || e == "--X")
      x--;
  }
  cout << x << '\n';
}
