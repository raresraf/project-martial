/******************************************
 *  Author : Debopriya Deb Roy
 *  Created On : Wed Jul 01 2020
 *  File : c++ file
 *******************************************/

/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem that detects "dangerous" situations in a string.
 * A situation is considered dangerous if a team has 7 or more consecutive identical characters (e.g., '0000000' or '1111111').
 * The program reads a string representing a sequence of players (0s and 1s) and determines if a dangerous situation exists.
 * Algorithm: Iterative string traversal with consecutive character counting.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  // Declares 'i' as an index (initialized to 1 for string traversal),
  // and other unused integer variables (j, k, l, m, n) which are common in competitive programming templates.
  int i = 1, j, k, l, m, n;
  // Declares 's' to store the input string.
  string s;
  // Reads the input string from standard input.
  cin >> s;
  
  // Stores the initial size of the string to control the while loop iterations.
  int size = s.size();
  // 'count' tracks consecutive identical characters, initialized to 1 for the first character.
  int count = 1;
  // 'check' stores the character currently being checked for consecutive occurrences, initialized with the first character of the string.
  int check = s[0];
  
  /**
   * Block Logic: Iterates through the string to find sequences of identical characters.
   * Invariant: At the start of each iteration, 'size' represents remaining characters to check,
   * 'count' is the current consecutive count of 'check' character ending at s[i-1],
   * 'i' is the index of the character currently being evaluated.
   * Pre-condition: 's' contains at least one character.
   */
  while (size--) {
    // Conditional Logic: Checks if the current character is the same as the 'check' character.
    if (check == s[i]) {
      count++; // Increments consecutive count if characters match.
      // Conditional Logic: If 7 consecutive identical characters are found, breaks the loop early.
      if (count == 7)
        break;
    } else {
      // If characters do not match, resets the consecutive count to 1 for the new character.
      count = 1;
      // Updates 'check' to the new character.
      check = s[i];
    }
    i++; // Moves to the next character in the string.
  }
  
  // Final Conditional Logic: Checks if a "dangerous" situation (7 consecutive identical characters) was found.
  if (count == 7)
    cout << "YES"; // Prints "YES" if found.
  else {
    cout << "NO"; // Prints "NO" if not found.
  }

  return 0; // Indicates successful program execution.
}