/**
 * @file 96A.CPP
 * @brief Determines if a football team has a "dangerous" situation.
 *
 * This program reads a string representing a sequence of players from two
 * different teams (represented by '0' and '1'). It determines if there is a
 * continuous sequence of 7 or more players from the same team.
 *
 * @see https://codeforces.com/problemset/problem/96/A
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string c;
  int count = 1;
  cin >> c;
  /**
   * @brief Main processing loop.
   *
   * Iterates through the string of players to find a continuous sequence of
   * 7 identical characters.
   */
  for (int i = 1; i < c.length(); ++i) {
    /**
     * @brief Sequence counting logic.
     *
     * If the current player is the same as the previous one, the counter is
     * incremented. If the counter reaches 7, it's a "dangerous" situation,
     * and the program prints "YES" and exits. If the current player is
     * different, the counter resets to 1.
     */
    if (c[i] == c[i - 1]) {
      count++;
      if (count == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else
      count = 1;
  }
  // If the loop completes without finding a sequence of 7, it's not a
  // dangerous situation.
  cout << "NO" << endl;

  return 0;
}
