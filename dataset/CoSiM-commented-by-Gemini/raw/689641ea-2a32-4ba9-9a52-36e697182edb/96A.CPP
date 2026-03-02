// Author : Mosiur Rahman Sweet

#include <bits/stdc++.h>
using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem from Codeforces (Problem 96A).
 * @details The program determines if a football situation is "dangerous," which occurs
 * if there are at least seven consecutive players from the same team ('0' or '1').
 *
 * Algorithm:
 * This solution manually iterates through the string to count consecutive identical characters.
 * It maintains a `counter` for the current run of identical players. The loop starts from
 * the second character and compares it with the previous one.
 * - If the characters are the same, the `counter` is incremented.
 * - If they differ, the `counter` is reset to 1.
 * At each step, the program checks if the `counter` has reached 7. If it does, a flag is
 * set, and the loop terminates immediately. Finally, based on the flag's value, the
 * program prints "YES" or "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string, as it requires a
 * single pass through the string.
 * Space Complexity: O(L) to store the input string.
 */
int main() {
  string s;
  cin >> s;
  // `counter` tracks the length of the current run of identical characters.
  int counter = 1;
  // `flag` is set to 1 if a dangerous situation is found.
  int flag = 0;

  /**
   * @brief Loop through the string to find a run of 7 identical characters.
   * The loop starts from the second character to allow comparison with the previous one.
   * The condition `s[i]` is a C-style check for the end of the string (null terminator).
   */
  for (int i = 1; s[i]; i++) {
    // If the current character is the same as the previous one, extend the run.
    if (s[i] == s[i - 1])
      counter++;
    else
      // Otherwise, reset the counter for a new run.
      counter = 1;

    // Check if the current run has reached a length of 7.
    if (counter == 7) {
      flag = 1;
      break;
    }
  }
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
