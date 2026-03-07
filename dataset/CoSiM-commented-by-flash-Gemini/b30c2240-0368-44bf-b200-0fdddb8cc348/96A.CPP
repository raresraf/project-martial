// 96A.cpp
// A. Football
// Author: Ikki
// May 31, 2015

/**
 * @file 96A.CPP
 * @brief Solution to the "Football" competitive programming problem (Codeforces 96A).
 * This program determines if a team is in a "dangerous" situation. A dangerous situation
 * is defined as having 7 or more consecutive players of the same team (represented by
 * identical characters, e.g., '0000000' or '1111111') in a sequence.
 * Algorithm: Iterative string traversal with consecutive character counting.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream> // For input/output operations.
#include <string>   // For string manipulation.
using namespace std;

int main() {
  string str; // Declares 'str' to store the input string representing the player sequence.
  // Reads the input string from standard input.
  cin >> str;
  
  // Initializes 'cnt' to 1, representing the count of consecutive identical characters.
  // It starts at 1 because the first character is considered as one instance of itself.
  int cnt = 1;
  
  /**
   * Block Logic: Iterates through the string starting from the second character to check for consecutive identical characters.
   * Invariant: At the start of each iteration, 'i' is the current character's index (1-based relative to the start of the string),
   * and 'cnt' holds the count of consecutive identical characters ending at `str[i-1]`.
   */
  for (int i = 1; i < str.size(); i++) {
    // Conditional Logic: Checks if the current character is the same as the previous character.
    if (str[i] == str[i - 1]) {
      cnt++; // Increments the consecutive count.
      // Conditional Logic: If 7 or more consecutive identical characters are found.
      if (cnt >= 7) {
        cout << "YES" << endl; // Prints "YES" as a dangerous situation is found.
        return 0;              // Terminates the program early.
      }
    } else {
      // If the current character is different from the previous one, reset the consecutive count to 1.
      cnt = 1;
    }
  }
  // If the loop completes without finding 7 or more consecutive identical characters.
  cout << "NO" << endl; // Prints "NO".
  
  return 0; // Indicates successful program execution.
}