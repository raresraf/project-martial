
// 56680198	Practice:havoccboi	96A - 10	GNU C++17	Accepted
// 62 ms	16 KB	2019-07-07 20:54:01	2019-07-07 20:54:01

#include <bits/stdc++.h>

using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A, "Football".
 * @details The program determines if a situation is "dangerous" by checking for
 * the presence of at least seven consecutive players from the same team.
 *
 * Algorithm:
 * The program iterates through the input string of '0's and '1's, keeping a
 * running count of consecutive identical characters. If two adjacent characters
 * are the same, the counter is incremented. If they differ, the counter is
 * reset to 1. As soon as the counter reaches 7, a boolean flag is set, and
 * the loop terminates immediately. The final output is based on the state of
 * this flag.
 *
 * Time Complexity: O(N), where N is the length of the input string. The loop
 * may terminate early, but the worst case is a single pass.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  string s;
  cin >> s;

  // `c` tracks the count of the current run of identical characters.
  int c = 1;
  // `flag` is set to true once a dangerous situation is confirmed.
  bool flag = false;

  // Optimization: A dangerous situation is impossible with fewer than 7 players.
  if ((int)s.size() < 7) {
    cout << "NO" << endl;
  } else {
    /**
     * @brief Iterate through the players to find a run of 7.
     */
    for (int i = 0; i < (int)s.size() - 1; i++) {
      if (s[i] == s[i + 1]) {
        c++;
      } else {
        // If the run is broken, reset the counter.
        c = 1;
      }

      // If a run of 7 is found, set the flag and exit the loop early.
      if (c == 7) {
        flag = true;
        break;
      }
    }

    // Output the result based on whether the flag was set.
    if (flag == true)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
}
