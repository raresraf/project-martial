/**
 * @file 96A.CPP
 * @brief Solution for the "Football" problem.
 *
 * @problem A situation in a football match is represented by a string of '0's and '1's.
 * A situation is "dangerous" if there are at least 7 consecutive players of the
 * same team. The task is to determine if the given situation is dangerous.
 *
 * @algorithm The code iterates through the input string, keeping two separate
 * counters: `n` for consecutive '1's and `m` for consecutive '0's. When a character
 * is encountered, the corresponding counter is incremented. The counter for the
 * other character is reset only when the next character changes. If either counter
 * reaches 7, the situation is dangerous.
 *
 * @implementation_note The code has flaws:
 * 1. It contains a redundant initial loop to count the total number of '0's and '1's.
 * 2. The main logic is wrapped in an `if (a > 0 && b > 0)` check, which is buggy.
 *    It will fail to detect a dangerous situation in a string of all '0's or all '1's
 *    (e.g., "0000000"), as it will not enter the main checking loop and will print nothing.
 *
 * @time_complexity O(N), where N is the length of the string, due to the linear scan.
 * @space_complexity O(N) to store the input string.
 */
#include <iostream>
#include <string> // Included for clarity

using namespace std;

int main() {
  int len, i, n = 0, m = 0, a = 0, b = 0;
  string str;
  cin >> str;
  len = str.length();

  // This first loop to count total 'a' and 'b' is redundant for the problem.
  for (i = 0; i < len; i++) {
    if (str[i] == '0')
      a++;
    if (str[i] == '1')
      b++;
  }

  // BUG: This condition makes the program fail for inputs containing only '0's or only '1's.
  // For an input like "0000000", `b` would be 0, this block is skipped, and nothing is printed.
  if (a > 0 && b > 0) {
    for (i = 0; i < len; i++) {
      // Logic for counting consecutive '1's
      if (str[i] == '1') {
        n++; // Increment consecutive '1's count
        if (n == 7) {
          cout << "YES";
          return 0;
        }
        // If the next character is different, reset the counter.
        if (str[i + 1] == '0') {
          n = 0;
        }
      }

      // Logic for counting consecutive '0's
      if (str[i] == '0') {
        m++; // Increment consecutive '0's count
        if (m == 7) {
          cout << "YES";
          return 0;
        }
        // If the next character is different, reset the counter.
        if (str[i + 1] == '1') {
          m = 0;
        }
      }
    }
    // If the loop finishes without finding 7 consecutive characters.
    cout << "NO";
  }
  return 0;
}
