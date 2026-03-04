/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 *
 * This program determines if a "dangerous" situation has occurred in a football
 * match, represented by a string of '0's and '1's. A dangerous situation is
 * defined as having at least seven consecutive players of the same team.
 *
 * @algorithm Linear scan with a running counter.
 * @time_complexity O(N), where N is the length of the input string, as it
 *                  involves a single pass through the string.
 * @space_complexity O(N) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {

  string s;
  cin >> s;
  int counter = 1;
  /**
   * @brief Scans the string for consecutive identical characters.
   * This loop iterates through the string, maintaining a `counter` for the
   * current run of identical characters. If the next character is the same,
   * the counter is incremented. If it's different, the counter is reset to 1.
   * If the counter ever reaches 7, it indicates a dangerous situation, and
   * the program prints "YES" and terminates early.
   */
  for (int i = 1; i < s.length(); i++) {

    if (s[i - 1] == s[i]) {
      counter++;
      if (counter == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      counter = 1;
    }
  }

  // An additional check after the loop in case the string is shorter than 7
  // or the sequence of 7 is at the very end. The check inside the loop is
  // sufficient, making this redundant.
  if (counter == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}