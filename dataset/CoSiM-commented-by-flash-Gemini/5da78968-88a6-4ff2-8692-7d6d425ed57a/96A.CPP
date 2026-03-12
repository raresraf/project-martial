/**
 * @5da78968-88a6-4ff2-8692-7d6d425ed57a/96A.CPP
 * @brief This program solves problem 96A, commonly known as "Football".
 * It analyzes a sequence of players (represented by '0's and '1's in a string)
 * to detect a "dangerous situation". A dangerous situation occurs if there are
 * 7 or more consecutive players of the same type.
 *
 * Algorithm: Single pass linear scan with two counters for consecutive identical characters.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) for storing counters.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string.

using namespace std; // Uses the standard namespace.

#define ll long long // Defines 'll' as a shorthand for long long (unused in this specific solution).

int main() {
  string s; // Declares a string variable to store the input sequence of players.
  cin >> s; // Reads the sequence string from standard input.
  int z = 0; // Initializes 'z' to count consecutive '0's.
  int u = 0; // Initializes 'u' to count consecutive '1's.
  
  /**
   * Block Logic: Iterates through each character of the input string, updating
   * the counters for consecutive '0's and '1's. It checks for a dangerous situation
   * after processing each character.
   * Invariant: `z` and `u` accurately reflect the current consecutive counts of '0's and '1's.
   */
  for (int i = 0; i < s.length(); i++) { // Loops through each character of the string.
    if (s[i] == '0') { // Inline: If the current character is '0'.
      z += 1; // Increments the counter for consecutive '0's.
      u = 0;  // Resets the counter for consecutive '1's.
    } else { // Inline: If the current character is '1'.
      z = 0;  // Resets the counter for consecutive '0's.
      u += 1; // Increments the counter for consecutive '1's.
    }
    /**
     * Block Logic: Checks if a dangerous situation has occurred.
     * Pre-condition: `z` and `u` have been updated for the current character.
     * Invariant: If either `z` or `u` is 7 or more, "YES" is printed, and the program terminates.
     */
    if (z >= 7 || u >= 7) { // If there are 7 or more consecutive '0's OR 7 or more consecutive '1's.
      cout << "YES" << "\n"; // Prints "YES" followed by a newline.
      return 0;              // Terminates the program immediately.
    }
  }
  // If the loop completes without finding a dangerous situation.
  cout << "NO" << "\n"; // Prints "NO" followed by a newline.
  return 0;              // Indicates successful program termination.
}
