/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: "Football".
 * Determines if a "dangerous situation" occurs in a football game, defined as
 * having 7 or more consecutive players of the same team ('0' or '1') in a sequence.
 * Algorithm: Iterates through the player sequence, maintaining a counter for consecutive
 * identical characters. If the counter reaches 7, a dangerous situation is found.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream> // Required for input/output operations (cin, cout, endl).
#include <string>   // Required for string manipulation.
using namespace std;

/**
 * @brief Main function to solve the "Football" problem.
 * Reads a string representing a sequence of players and determines if it contains
 * a dangerous situation (7 or more consecutive identical players).
 */
int main() {
  string s; // Input string representing the sequence of players.
  cin >> s; // Reads the input string.
  
  int counter = 1; // Counter for consecutive identical characters, initialized to 1 for the first character.

  /**
   * Block Logic: Iterates through the string from the second character to check for consecutive identical players.
   * Pre-condition: 's' contains the player sequence. 'counter' is 1.
   * Invariant: 'counter' accurately tracks the current streak of identical characters ending at `s[i]`.
   *            The loop may terminate early if a dangerous situation is found.
   */
  for (int i = 1; i < s.length(); i++) {
    /**
     * Block Logic: Compares the current character with the previous one.
     * If they are the same, increments the counter. If the counter reaches 7,
     * a dangerous situation is confirmed and the program exits.
     */
    if (s[i - 1] == s[i]) {
      counter++; // Increment counter for consecutive identical characters.
      if (counter == 7) {
        cout << "YES" << endl; // Dangerous situation found.
        return 0;              // Terminate the program early.
      }
    } else {
      // If characters are different, reset the counter for a new streak.
      counter = 1;
    }
  }

  /**
   * Block Logic: Final check if a dangerous situation exists after the loop completes.
   * This handles cases where 7 or more consecutive identical characters are found
   * exactly at the end of the string, and the `if (counter == 7)` inside the loop
   * did not trigger a `return 0`.
   */
  if (counter == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0; // Indicates successful execution if no dangerous situation was found or already printed.
}
