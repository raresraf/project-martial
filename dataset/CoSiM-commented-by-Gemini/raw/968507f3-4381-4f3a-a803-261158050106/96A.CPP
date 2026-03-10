/**
 * @file 96A.CPP
 * @brief Solution for a competitive programming problem.
 * This program determines if a given binary string contains a contiguous
 * sequence of at least 7 identical characters ('0' or '1').
 *
 * @algorithm Single-pass linear scan.
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(N) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;

  cin >> s;

  int count_0 = 0, count_1 = 0, max_0 = 0, max_1 = 0;

  /**
   * @block
   * @brief Iterates through the string to find the longest run of '0's and '1's.
   *
   * This loop scans the input string character by character. It maintains two
   * counters, `count_0` and `count_1`, to track the current consecutive
   * occurrences of '0's and '1's, respectively. When a different character
   * is encountered, the corresponding counter is reset. The maximum
   * consecutive counts are stored in `max_0` and `max_1`.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '0') {
      count_1 = 0;
      count_0++;
      if (max_0 < count_0)
        max_0 = count_0;
      //	cout << max_0 << endl;
    } else {
      count_0 = 0;
      count_1++;
      if (max_1 < count_1)
        max_1 = count_1;
      //	cout << max_1 << endl;
    }
  }

  /**
   * @block
   * @brief Checks if any of the longest runs meet the threshold.
   *
   * If the maximum length of consecutive '0's or '1's is 7 or more,
   * it prints "YES", indicating that the condition is met. Otherwise, it prints "NO".
   */
  if ((max_0 >= 7) || (max_1 >= 7))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
