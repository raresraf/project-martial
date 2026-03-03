/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 * This program determines if a "dangerous" situation has occurred in a football match.
 * A situation is considered dangerous if a player from one team has the ball for
 * 7 consecutive minutes. The input is a string of '0's and '1's, representing
 * which team has the ball each minute.
 * Algorithm: Single pass iteration to count consecutive identical characters.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) - to store the input string.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  string players; // Declares a string to store the sequence of players having the ball.
  cin >> players; // Reads the input string from standard input.
  
  // Initializes counters for consecutive '0's and '1's.
  // 'result' stores "NO" by default, changing to "YES" if a dangerous situation is found.
  int zeros_count, ones_count; // Renamed 'zeros' to 'zeros_count', 'ones' to 'ones_count' for clarity.
  string result = "NO";
  zeros_count = ones_count = 0; // Initialize both counters to zero.
  
  /**
   * Block Logic: Iterates through the 'players' string to count consecutive occurrences of '0' or '1'.
   * If a sequence of 7 identical characters is found, the 'result' is set to "YES".
   * Invariant: `zeros_count` and `ones_count` track the current consecutive count of '0's and '1's,
   *            resetting each other when the character changes.
   */
  for (int i = 0; i < players.length(); i++) { // Loop through each character of the string.
    if (players[i] == '1') {
      ones_count++;   // Increment count for '1'.
      zeros_count = 0; // Reset count for '0'.
    }
    if (players[i] == '0') {
      zeros_count++;  // Increment count for '0'.
      ones_count = 0; // Reset count for '1'.
    }
    
    // Checks if either team has had the ball for 7 or more consecutive minutes.
    if ((zeros_count == 7) || (ones_count == 7))
      result = "YES"; // A dangerous situation is detected.
  }
  
  // Prints the final result ("YES" or "NO") followed by a newline.
  cout << result << "\n";
  
  return 0; // Indicates successful program execution.
}
