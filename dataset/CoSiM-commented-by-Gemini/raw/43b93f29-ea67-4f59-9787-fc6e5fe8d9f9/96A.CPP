/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 *
 * @algorithm
 * The problem defines a "dangerous" situation as having at least 7 consecutive
 * players from the same team ('0' or '1'). This program checks if such a
 * sequence exists in the input string.
 *
 * This solution uses a dual-counter approach. It maintains two separate counters,
 * one for consecutive '0's and one for consecutive '1's.
 *
 * The algorithm is as follows:
 * 1. Read the input string `s`.
 * 2. Initialize `zero_count` and `one_count` to 0.
 * 3. Iterate through each character of the string.
 * 4. If the character is '0', increment `zero_count` and reset `one_count` to 0.
 * 5. If the character is '1', increment `one_count` and reset `zero_count` to 0.
 * 6. After updating the counters in each iteration, check if either count has
 *    reached 7.
 * 7. If a dangerous sequence is found, set a flag and break the loop immediately.
 * 8. Print "YES" or "NO" based on whether the flag was set.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string, due to a single pass.
 * - Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int len, zero = 0, one = 0, flag = 0;
  string s;
  cin >> s;
  len = s.length();
  /**
   * @brief Iterates through the player string, tracking consecutive runs of '0's and '1's.
   */
  for (int i = 0; i < len; i++) {
    // If the current player is '0', increment the zero counter and reset the one counter.
    if (s[i] == '0')
      zero++;
    else
      zero = 0;
    // If the current player is '1', increment the one counter and reset the zero counter.
    if (s[i] == '1')
      one++;
    else
      one = 0;
    /**
     * @brief Check if a dangerous situation has occurred.
     * If either counter reaches 7, set the flag and exit the loop.
     */
    if (zero == 7 || one == 7) {
      flag = 1;
      break;
    }
  }
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
