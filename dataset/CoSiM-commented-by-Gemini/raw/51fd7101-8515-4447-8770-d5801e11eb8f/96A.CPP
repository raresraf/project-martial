/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 * @details The problem is to determine if a given string of '0's and '1's,
 * representing players, contains a "dangerous" situation. A situation is
 * dangerous if there are 7 or more consecutive players of the same team.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Initialize a counter `c` to 1 to track the length of the current run
 *    of identical characters.
 * 3. Initialize a boolean flag `f` to false, which will be set to true if a
 *    dangerous situation is found.
 * 4. Iterate through the string from the second character to the end.
 * 5. In each iteration, compare the current character with the previous one.
 *    a. If they are the same, increment the run counter `c`. If `c` reaches 7,
 *       a dangerous situation is found. Set the flag `f` to true and exit the
 *       loop.
 *    b. If they are different, the run is broken. Reset the counter `c` to 1.
 * 6. After the loop, if the flag `f` is true, print "YES". Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the single
 * pass through the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s; // To store the input string of players.
  getline(cin, s);
  
  int len = s.size();
  // 'c' is the counter for the current consecutive run of identical characters.
  int c = 1;
  // 'f' is a flag to indicate if a dangerous situation has been found.
  bool f = false;

  /**
   * @block
   * @brief Iterate through the string to find a run of 7 identical characters.
   */
  for (int i = 1; i < len; i++) {
    // Compare the current character with the previous one.
    if (s[i] == s[i - 1]) {
      // If they are the same, increment the run counter.
      c++;
      // Check if the run has reached the dangerous threshold.
      if (c == 7) {
        f = true; // Set the flag and break, as the condition is met.
        break;
      }
    } else {
      // If the characters are different, the run is broken. Reset the counter.
      c = 1;
      // Resetting 'f' here is redundant but doesn't harm the logic.
      f = false; 
    }
  }

  // Print the result based on whether the flag was set.
  printf("%s\n", f ? "YES" : "NO");

  return 0;
}
