/**
 * @file 96A.CPP
 * @brief Determines if a football "situation" is dangerous based on consecutive players of the same team.
 *   A situation is considered dangerous if there are 7 or more consecutive players
 *   of the same team (represented by '0' or '1').
 * * Algorithm: Single-Pass Consecutive Count.
 *   The program iterates through the sequence of players, maintaining a count of
 *   consecutive identical players. If this count ever reaches 7, the situation is
 *   deemed dangerous.
 * Time Complexity: O(L) where L is the length of the input string representing the players.
 * Space Complexity: O(L) for storing the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to analyze the football situation for danger.
 * Functional Utility: Handles input, processes the player sequence, and determines if it's a dangerous situation.
 */
int main() {
  string s; // String representing the sequence of players.
  cin >> s;

  bool check = true; // Flag to track if the situation is NOT dangerous (true initially).
  char curr = s[0];  // Stores the character of the current consecutive player sequence.
  int count = 1;     // Counts consecutive identical players.

  /**
   * Block Logic: Iterates through the player sequence to find consecutive identical players.
   * Invariant: 'i' is the current player index, 'curr' is the character of the current sequence,
   *   and 'count' is the length of this sequence.
   */
  for (int i = 1; i < s.length(); i++) {
    /**
     * Condition: Checks if the current player is the same as the previous one in the sequence.
     * If true, increments the consecutive count.
     */
    if (s[i] == curr) {
      count++;
    } else {
      // If the player changes, reset the current player and restart the count.
      curr = s[i];
      count = 1;
    }

    /**
     * Condition: Checks if the count of consecutive identical players has reached 7.
     * If true, the situation is dangerous, set 'check' to false and break the loop.
     */
    if (count == 7) {
      check = false;
      break;
    }
  }

  /**
   * Block Logic: Outputs the result based on whether a dangerous situation was detected.
   * Pre-condition: 'check' flag indicates if 7 consecutive players were found.
   */
  if (check)
    cout << "NO"; // No dangerous situation found.
  else
    cout << "YES"; // Dangerous situation found.
}