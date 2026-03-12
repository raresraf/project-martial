/**
 * @564d0a88-8e05-43f8-b2b9-da3f2d5a899c/96A.CPP
 * @brief This program determines if a "dangerous situation" exists in a sequence of players.
 * A dangerous situation is defined as having 7 consecutive players of the same type
 * (represented by identical characters, e.g., '0000000' or '1111111').
 *
 * Algorithm: Single pass scan with consecutive character counter.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) for storing counters and input string.
 */

#include <iostream> // Includes the standard input/output stream library.
#include <string>   // Includes the string library for using std::string.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  string s; // Declares a string variable to store the input sequence of players.
  cin >> s; // Reads the sequence string from standard input.

  int time = 1; // Initializes a counter for consecutive identical characters.
                // It starts at 1 because the first character itself starts a sequence of length 1.
  /**
   * Block Logic: Iterates through the string to detect sequences of 7 identical characters.
   * Invariant: 'time' accurately tracks the length of the current consecutive identical character sequence.
   */
  for (size_t i = 1; i < s.length(); ++i) { // Loops from the second character to the end of the string.
    /**
     * Block Logic: Checks if the current character is the same as the previous one.
     * Pre-condition: 'i' is a valid index within the string, and 's[i-1]' is accessible.
     * Invariant: 'time' is incremented if characters match, or reset if they differ.
     */
    if (s[i] == s[i - 1]) { // Compares the current character with the preceding character.
      time += 1; // Inline: Increments the consecutive character counter.
      /**
       * Block Logic: Checks if a dangerous situation (7 consecutive identical characters) has been found.
       * Pre-condition: 'time' has just been updated after a match.
       * Invariant: If 'time' is 7, the program immediately outputs "YES" and terminates.
       */
      if (time == 7) {
        cout << "YES" << endl; // Output indicating a dangerous situation.
        return 0;             // Terminates the program.
      }
    } else {
      time = 1; // Inline: Resets the counter to 1 as a new sequence starts with the current character.
    }
  }

  // If the loop completes without finding 7 consecutive identical characters.
  cout << "NO" << endl; // Output indicating no dangerous situation was found.
  return 0;             // Indicates successful program termination.
}