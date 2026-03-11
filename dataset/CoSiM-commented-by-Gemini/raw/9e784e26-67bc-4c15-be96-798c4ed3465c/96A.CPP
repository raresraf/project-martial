/**
 * @file 96A.CPP
 * @brief Determines if a string contains a sequence of 7 or more identical characters ('0' or '1').
 *
 * This program solves the "Football" problem, where a situation is considered
 * "dangerous" if there is a contiguous block of 7 or more players of the same
 * team. The teams are represented by '0' and '1'.
 *
 * @algorithm The program iterates through the input string, using two separate
 * counters (`z` for '0's and `o` for '1's) to track the length of the current
 * consecutive run of identical characters. When the character changes, the
 * corresponding counter is reset. If either counter reaches 7 at any point,
 * the loop is terminated, and the situation is declared "dangerous".
 *
 * @note The implementation has some quirks. The counter for '1's (`o`) is
 * initialized to 1, while the counter for '0's (`z`) is 0, which is asymmetric
 * but works for this problem. The loop condition `while (s[i] != '\0')` is a
 * C-style approach to string iteration; a C++ `for` loop with `s.length()` would
 * be more idiomatic.
 *
 * @time_complexity O(N), where N is the length of the string, as it performs a
 * single pass.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // z: counter for consecutive '0's. o: counter for consecutive '1's.
  // The initialization of 'o' to 1 is unusual but does not break the logic.
  int z = 0, o = 1;
  string s;
  cin >> s;
  int i = 0;

  /**
   * @brief Loop to iterate through the string and find a run of 7.
   * @invariant `z` and `o` hold the current run lengths of '0's and '1's
   * ending at the previous character.
   */
  while (s[i] != '\0') {
    if (s[i] == '0') {
      z++;    // Increment '0' counter.
      o = 0;  // Reset '1' counter.
      if (z == 7)
        break; // Dangerous situation found.
    } else {
      o++;    // Increment '1' counter.
      z = 0;  // Reset '0' counter.
      if (o == 7)
        break; // Dangerous situation found.
    }
    i++;
  }
  
  /**
   * @brief Check if a dangerous situation was found.
   * @pre The loop has completed, either by finding a run of 7 or by reaching
   * the end of the string.
   */
  if (z == 7 || o == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
