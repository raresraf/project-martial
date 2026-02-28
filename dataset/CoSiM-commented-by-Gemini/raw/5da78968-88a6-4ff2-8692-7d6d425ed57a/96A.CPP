/**
 * @file 96A.CPP
 * @brief Solution for the "Football" problem from Codeforces.
 * @details The problem is to determine if a situation is "dangerous" in a football match.
 * A situation is dangerous if there is a sequence of at least 7 consecutive players
 * from the same team. The teams are represented by '0's and '1's in a string.
 *
 * @algorithm Single-Pass Consecutive Counting
 * 1. Read the input string representing the player sequence.
 * 2. Initialize two counters to zero: one for consecutive '0's and one for '1's.
 * 3. Iterate through the string character by character.
 * 4. If the current character is '0', increment the '0's counter and reset the '1's counter.
 * 5. If the current character is '1', increment the '1's counter and reset the '0's counter.
 * 6. After each update, check if either counter has reached 7.
 * 7. If a counter reaches 7, the condition is met. Print "YES" and terminate immediately.
 * 8. If the loop completes without any counter reaching 7, print "NO".
 *
 * @complexity
 * - Time: O(N), where N is the length of the string, as it's processed in a single pass.
 * - Space: O(N), to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

// Macro for long long int type, though not used in this solution.
#define ll long long

int main() {
  // s: stores the sequence of players as a string of '0's and '1's.
  string s;
  cin >> s;
  
  // z: counter for consecutive '0's.
  int z = 0;
  // u: counter for consecutive '1's.
  int u = 0;

  /**
   * @brief Iterate through the string to find a consecutive sequence of 7.
   */
  // Invariant: At the start of each iteration, `z` and `u` hold the counts of
  // consecutive '0's or '1's ending at index `i-1`.
  for (int i = 0; i < s.length(); i++) {
    // Pre-condition: Check if the current player is from team '0'.
    if (s[i] == '0') {
      z += 1; // Increment the '0's streak.
      u = 0;  // Break the '1's streak.
    } else {
      z = 0;  // Break the '0's streak.
      u += 1; // Increment the '1's streak.
    }
    
    /**
     * @brief Check for a dangerous situation after each player.
     * If a streak of 7 or more is found, print "YES" and exit immediately
     * as the condition has been met.
     */
    if (z >= 7 || u >= 7) {
      cout << "YES"
           << "\n";
      return 0;
    }
  }

  // If the loop completes without returning, no streak of 7 was found.
  cout << "NO"
       << "\n";
       
  return 0; // Indicate successful execution.
}
