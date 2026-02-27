/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 * @details The problem is to determine if a "dangerous" situation has occurred,
 * which is defined as seven or more consecutive players of the same team.
 * The teams are represented by '0' and '1' in a string.
 *
 * @algorithm
 * The algorithm iterates through the string, keeping a running count of
 * consecutive identical characters.
 * 1. Read the input string.
 * 2. Initialize a counter for the current run to 1.
 * 3. Loop through the string from the second character to the end.
 *    a. Compare the current character with the previous one.
 *    b. If they are the same, increment the run counter. If the counter
 *       reaches 7, a dangerous situation is found; print "YES" and terminate.
 *    c. If they are different, reset the run counter back to 1.
 * 4. If the loop completes without the counter ever reaching 7, no dangerous
 *    situation occurred; print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the single pass.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  // 'time' is used as a counter for the length of the current consecutive run.
  int time = 1;
  
  /**
   * @block
   * @brief Iterate through the string to find a run of 7 identical characters.
   */
  for (size_t i = 1; i < s.length(); ++i) {
    if (s[i] == s[i - 1]) {
      // If the character is the same as the previous one, extend the run.
      time += 1;
      // If the run reaches 7, the condition is met.
      if (time == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      // If the character changes, reset the run counter to 1.
      time = 1;
    }
  }

  // If the loop completes, no run of 7 was found.
  cout << "NO" << endl;
  return 0;
}
