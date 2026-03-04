/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 *
 * This program determines if a string representing a sequence of football players
 * (represented by '0's and '1's) contains a "dangerous" situation. A dangerous
 * situation is defined as a sequence of 7 or more identical consecutive
 * characters.
 *
 * @algorithm Linear scan with running counters.
 * @time_complexity O(N), where N is the length of the input string. The program
 *                  iterates through the string once.
 * @space_complexity O(N) to store the input string and its C-style character
 *                   array copy.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Optimizes I/O operations.
   * `ios::sync_with_stdio(0)` disables synchronization with C-style I/O.
   * `cin.tie(0)` and `cout.tie(0)` untie cin from cout. These are common
   * optimizations in competitive programming for faster I/O.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string str;
  long long countt = 0, counti = 0;
  getline(cin, str);

  int n = str.length();
  char charr[n + 1];
  strcpy(charr, str.c_str());

  /**
   * @brief Scans the string to find consecutive identical characters.
   * This loop iterates through the string, character by character. It maintains
   * two counters, `counti` for '0's and `countt` for '1's. When a character
   * is encountered, the corresponding counter is incremented and the other is
   * reset. The loop breaks if either counter reaches 7.
   */
  for (int i = 0; i < n + 1; i++) {
    if (charr[i] == '0') {
      countt = 0;
      counti = counti + 1;
      if (counti == 7) {
        break;
      }
    } else if (charr[i] == '1') {
      counti = 0;
      countt = countt + 1;
      if (countt == 7) {

        break;
      }
    }
  }

  /**
   * @brief Outputs the result.
   * If either of the counters reached 7, it means a dangerous situation was
   * found, and "YES" is printed. Otherwise, "NO" is printed.
   */
  if (counti == 7 || countt == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}