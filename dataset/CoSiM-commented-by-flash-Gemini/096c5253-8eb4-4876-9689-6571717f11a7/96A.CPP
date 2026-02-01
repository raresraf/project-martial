/**
 * @file 96A.CPP
 * @brief This program determines if a "dangerous" situation exists in a football match
 * based on a string representation of player positions. A situation is dangerous
 * if at least 7 players of the same team are standing consecutively.
 * Algorithm: Iterative scan with a counter for consecutive identical players.
 * Time Complexity: O(L) where L is the length of the input string, as it performs a single pass.
 * Space Complexity: O(1) beyond input storage.
 */

#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for string manipulation.

using namespace std; // Uses the standard namespace to avoid prefixing standard library elements with std::

int main() {
  string str; // Declares a string variable to store the player positions.
  cin >> str; // Functional Utility: Reads the input string representing player positions.

  long long max_consecutive = 0; // Stores the maximum count of consecutive identical players found so far.
  long long current_consecutive = 0; // Stores the current count of consecutive identical players.

  /*
   * Block Logic: Iterates through the string to find the longest sequence of consecutive identical players.
   * Invariant: 'current_consecutive' tracks the length of the current run of identical characters.
   *            'max_consecutive' stores the maximum run length encountered up to the current position.
   * This loop compares adjacent characters to detect and count consecutive players of the same team.
   */
  for (int i = 0; i < str.length() - 1; i++) {
    // Checks if the current player and the next player are from the same team.
    if (str[i] == str[i + 1]) {
      current_consecutive++; // Increments the counter for consecutive players.
      // Updates the overall maximum consecutive count if the current run is longer.
      // Pre-condition: 'current_consecutive' represents the length of the current run of identical characters starting from 'str[i-current_consecutive]'.
      // Invariant: 'max_consecutive' always holds the maximum number of consecutive identical characters found so far in the string.
      if (current_consecutive + 1 > max_consecutive) {
        max_consecutive = current_consecutive + 1;
      }
    } else {
      // If adjacent players are from different teams, reset the current consecutive counter.
      current_consecutive = 0;
    }
  }

  // Block Logic: Determines if the situation is dangerous based on the maximum consecutive count.
  // Pre-condition: 'max_consecutive' holds the length of the longest sequence of identical players.
  // A situation is dangerous if there are at least 7 consecutive players (max_consecutive > 6).
  if (max_consecutive > 6) {
    cout << "YES" << endl; // Prints "YES" if the situation is dangerous.
  } else {
    cout << "NO" << endl;  // Prints "NO" otherwise.
  }
  return 0; // Indicates successful program execution.
}