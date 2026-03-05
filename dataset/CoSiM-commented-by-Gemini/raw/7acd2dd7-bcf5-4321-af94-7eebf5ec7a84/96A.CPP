/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A, "Football".
 * @details This program determines if a situation is "dangerous" by checking if
 * there is a sequence of 7 or more identical characters (representing players
 * of the same team) in a given string.
 *
 * Algorithm:
 * The code uses a two-pass approach to solve the problem.
 * 1. It first iterates through the input string to find if there is a
 *    sub-string of seven or more consecutive '0's. It uses a counter that
 *    increments for each '0' and resets for each '1'.
 * 2. If such a sequence is found, a boolean flag is set to true.
 * 3. It then performs a second, identical pass through the string, this time
 *    counting consecutive '1's. If a run of seven or more is found, the same
 *    flag is set to true.
 * 4. Finally, it checks the status of the flag to determine if a dangerous
 *    situation occurred and prints "YES" or "NO" accordingly.
 *
 * Time Complexity: O(N), where N is the length of the string, as it performs
 * two linear scans.
 * Space Complexity: O(N) to store the input string.
 */
// http://codeforces.com/problemset/problem/96/A

#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int cnt = 0, sz = s.length();
  bool flag = false;

  /**
   * @brief First pass: Check for a dangerous situation with players of team '0'.
   * This loop counts consecutive '0's. If the count reaches 7, it sets the
   * `flag` to true. The counter resets whenever a '1' is encountered.
   */
  for (int i = 0; i < sz; i++) {
    if (s[i] == '0')
      cnt++;
    else {
      cnt = 0;
    }
    if (cnt >= 7)
      flag = true;
  }
  
  cnt = 0; // Reset counter for the next pass.

  /**
   * @brief Second pass: Check for a dangerous situation with players of team '1'.
   * This loop performs the same logic as the first, but for the character '1'.
   */
  for (int i = 0; i < sz; i++) {
    if (s[i] == '1')
      cnt++;
    else {
      cnt = 0;
    }
    if (cnt >= 7)
      flag = true;
  }

  /**
   * @brief If the flag was set in either of the passes, a dangerous situation
   * was found.
   */
  if (flag)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
