/**
 * @file 96A.CPP
 * @brief Solution for the "Football" problem on Codeforces.
 *
 * @details
 * This program determines if a situation in a football match is "dangerous".
 * A situation is dangerous if there is a sequence of at least 7 consecutive
 * players from the same team. The input is a string of '0's and '1's.
 * The program scans the string to find such a sequence.
 *
 * @section
 * Algorithm: Linear scan with a counter for consecutive characters.
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 *
 * @warning This implementation contains a bug. In the first iteration of the
 * while loop (when i=0), it attempts to access `str[i-1]`, which is an
 * out-of-bounds memory access and results in undefined behavior.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  // 'flag' counts the number of *consecutive matches*. A run of 7 identical
  // characters will result in flag reaching a value of 6.
  int flag = 0, i = 0;

  cin >> str;
  /**
   * @brief Main loop to scan the string for consecutive characters.
   * @warning The condition `i <= str.size()` is risky as it allows access to
   * `str[str.size()]` (the null terminator). More critically, on the first
   * iteration (i=0), `str[i-1]` is an out-of-bounds access.
   */
  while (i <= str.size()) {
    // Check if the current character matches the previous one.
    if (str[i] == str[i - 1]) {
      flag++;
      // If there have been 6 consecutive matches, it means 7 identical
      // characters have been found in a row.
      if (flag == 6) {
        cout << "YES" << endl;
        return 0; // Situation is dangerous, exit immediately.
      }
    }

    else
      // If the characters don't match, the streak is broken. Reset the counter.
      flag = 0;

    i++;
  }
  // If the loop completes, no dangerous situation was found.
  cout << "NO" << endl;

  return 0;
}
