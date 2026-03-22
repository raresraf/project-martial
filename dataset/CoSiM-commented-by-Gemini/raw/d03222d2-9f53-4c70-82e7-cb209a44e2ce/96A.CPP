/**
 * @file 96A.CPP
 * @brief Determines if a football situation is "dangerous".
 *
 * This program solves the "Football" problem. A situation is considered dangerous
 * if there are at least 7 consecutive players from the same team. The teams are
 * represented by '0's and '1's in an input string.
 *
 * @b Algorithm: Linear Scan with a Counter
 *   The code iterates through the string of players, maintaining a counter for the
 *   current run of consecutive identical players. If the counter ever reaches 7,
 *   the situation is deemed dangerous.
 *
 * @b Time_Complexity: O(L), where L is the length of the player string.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  cin >> input;

  int len = input.length();
  // counter: tracks the length of the current consecutive sequence of players.
  int counter = 1;
  // last: stores the team (0 or 1) of the previous player.
  // The character is converted to an integer by subtracting the ASCII value of '0'.
  int last = (int)input[0] - 48;
  
  /**
   * @brief Iterates through the players to find a dangerous situation.
   */
  for (int i = 1; i < len; i++) {
    // This structure checks if the current player continues the sequence or starts a new one.
    if ((int)input[i] - 48 == 0) {
      if (last == 0)
        counter++;
      else {
        // Reset counter for a new sequence of '0's.
        counter = 1;
        last = 0;
      }
    }
    if ((int)input[i] - 48 == 1) {
      if (last == 1)
        counter++;
      else {
        // Reset counter for a new sequence of '1's.
        counter = 1;
        last = 1;
      }
    }
    
    // If a sequence of 7 is found, the condition is met, and we can stop early.
    if (counter == 7)
      break;
  }

  // Check if the loop terminated because a dangerous situation was found.
  if (counter == 7)
    cout << "YES";
  else
    cout << "NO";
}
