/**
 * @689641ea-2a32-4ba9-9a52-36e697182edb/96A.CPP
 * @brief This program solves Codeforces problem 96A, "Football".
 * It analyzes a sequence of players (represented by '0's and '1's in a string)
 * to detect a "dangerous situation". A dangerous situation occurs if there are
 * 7 or more consecutive players of the same team (i.e., "0000000" or "1111111"
 * appears as a substring).
 *
 * Algorithm: Linear scan with a counter. It iterates through the string,
 * counting consecutive identical characters. If the count reaches 7, a flag is set.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
// Author : Mosiur Rahman Sweet

#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <string>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to detect a "dangerous situation" in a football game.
 * Reads a string representing player sequence and checks for 7 consecutive same-team players.
 * @return 0 on successful execution.
 */
int main() {
  string s; // Declares a string variable to store the input sequence of players.
  cin >> s; // Reads the player sequence from standard input.
  
  int counter = 1; // Counts consecutive identical characters; initialized to 1 for the first character.
  int flag = 0;    // Flag to indicate if a dangerous situation is found (0 for NO, 1 for YES).
  
  /**
   * Block Logic: Iterates through the string to count consecutive identical characters.
   * If a character is the same as the previous one, `counter` increments. Otherwise, it resets to 1.
   * If `counter` reaches 7, a dangerous situation is found, and the loop breaks.
   * Invariant: `counter` holds the current streak of identical characters; `flag` indicates discovery of a dangerous situation.
   */
  for (int i = 1; s[i]; i++) { // Loops from the second character (index 1) until the end of the string.
    // Pre-condition: Checks if the current character is the same as the previous one.
    if (s[i] == s[i - 1])
      counter++; // Increments counter for consecutive identical characters.
    else
      counter = 1; // Resets counter if characters are different.

    // Pre-condition: Checks if the counter for consecutive identical characters has reached 7.
    if (counter == 7) { 
      flag = 1; // Sets the flag to indicate a dangerous situation.
      break;    // Exits the loop early.
    }
  }
  
  /**
   * Block Logic: Prints the result based on the `flag` value.
   * Invariant: Prints "YES" if `flag` is 1, "NO" otherwise.
   */
  if (flag) // If `flag` is 1 (true).
    cout << "YES" << endl; // Prints "YES", followed by a newline.
  else // If `flag` is 0 (false).
    cout << "NO" << endl; // Prints "NO", followed by a newline.
  
  return 0; // Indicates successful program termination.
}
