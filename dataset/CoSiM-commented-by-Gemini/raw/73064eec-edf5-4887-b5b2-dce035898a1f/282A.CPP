/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 *
 * This program simulates a series of increment and decrement operations on a
 * single variable, 'x', initialized to zero. It reads the number of operations
 * and then processes each one, finally printing the final value of 'x'.
 * The implementation is highly compact.
 *
 * @algorithm Simple counting/simulation.
 * @time_complexity O(N), where N is the number of operations.
 * @space_complexity O(1), aside from the storage for the operation string.
 */
//#include <bits/stdc++.h>
// using namespace std;
#include <iostream>
#include <string>

int main() {
  // Fast I/O setup
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);

  int n, x = 0;
  std::cin >> n;
  std::string str;

  /**
   * @brief Processes all operations in a compact loop.
   * The loop condition `n--` decrements `n` in each iteration until it becomes 0.
   * Inside the loop, a string is read, and a ternary operator checks for a '+'
   * symbol at the first or third position to decide whether to increment or
   * decrement 'x'. This is a very condensed way to handle the logic.
   */
  for (int i = 0; i < n; n--) {
    std::cin >> str, (str[0] == '+') || (str[2] == '+') ? x++ : x--;
  }
  std::cout << x;

  return 0;
}