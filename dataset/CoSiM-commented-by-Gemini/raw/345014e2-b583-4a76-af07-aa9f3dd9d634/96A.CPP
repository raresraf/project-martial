/**
 * @file 96A.CPP
 * @brief Determines if there is a "dangerous" situation in a line of football players.
 *
 * This program solves the "Football" problem. A situation is considered "dangerous"
 * if there are at least seven consecutive players of the same team. The line of
 * players is represented by a string of '0's and '1's.
 *
 * Algorithm:
 * 1. Read the input string representing the players.
 * 2. Iterate through the string, keeping a count of consecutive identical characters.
 * 3. If the current character is the same as the previous one, increment the counter.
 * 4. If it's different, reset the counter to 1.
 * 5. If the counter ever reaches 7, the situation is dangerous. Print "YES" and terminate.
 * 6. If the loop completes without the counter reaching 7, print "NO".
 *
 * Time Complexity: O(N), where N is the number of players (length of the string),
 * as it requires a single pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  // flag: A boolean-like indicator to track if a dangerous situation was found.
  int flag = 0;
  // Read the string of players.
  cin >> s;
  
  // count: Tracks the number of consecutive identical players.
  int count = 1;

  /**
   * Block Logic: Iterate through the players to find consecutive groups.
   * This loop starts from the second player and compares each player with the one
   * immediately preceding them.
   */
  for (int i = 1; i < s.length(); i++) {
    // Invariant: 'count' holds the number of consecutive identical players ending at index i-1.
    if (s[i - 1] == s[i]) {
      // If the current player is from the same team as the previous, increment the streak.
      count++;
    } else {
      // If the team changes, reset the streak counter.
      count = 1;
    }

    // Pre-condition: Check if the streak has reached the dangerous threshold.
    if (count == 7) {
      cout << "YES";
      flag = 1; // Mark that a dangerous situation was found.
      break;    // No need to check further.
    }
  }

  // If the loop finished without finding a dangerous situation, print "NO".
  if (flag == 0) {
    cout << "NO";
  }
  return 0;
}
