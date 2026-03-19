/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: "Football".
 * Determines if a "dangerous situation" occurs in a football game, defined as
 * having 7 or more consecutive players of the same team ('0' or '1') in a sequence.
 * Algorithm: Iterate through the player sequence, maintaining counters for consecutive
 * '0's and '1's. If either counter reaches 7, a dangerous situation is found.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string and its character array copy.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream and string.
using namespace std;

/**
 * @brief Main function to solve the "Football" problem.
 * Reads a string representing a sequence of players and determines if it contains
 * a dangerous situation (7 or more consecutive identical players).
 */
int main() {
  // Functional Utility: Optimize C++ standard streams for faster input/output.
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string str;             // Input string representing the sequence of players.
  long long countt = 0;   // Counter for consecutive '1's (representing one team).
  long long counti = 0;   // Counter for consecutive '0's (representing the other team).
  
  getline(cin, str);      // Reads the entire line of input into 'str'.
  int n = str.length();   // Stores the length of the input string.
  
  // Creates a C-style character array and copies the content of 'str' into it.
  // The size 'n + 1' accounts for the null terminator.
  char charr[n + 1];
  strcpy(charr, str.c_str());

  /**
   * Block Logic: Iterates through the character array to detect consecutive identical players.
   * Invariant: 'counti' and 'countt' accurately reflect the current streak of '0's and '1's.
   *            The loop terminates early if a dangerous situation (7 consecutive) is found.
   */
  for (int i = 0; i < n + 1; i++) { // Loop goes up to n+1 to include the null terminator, though effectively checks up to n.
    /**
     * Block Logic: If the current character is '0', increment 'counti' and reset 'countt'.
     * Checks for a dangerous situation for '0's.
     */
    if (charr[i] == '0') {
      countt = 0;          // Reset consecutive '1's counter.
      counti = counti + 1; // Increment consecutive '0's counter.
      if (counti == 7) {   // If 7 consecutive '0's are found, it's a dangerous situation.
        break;             // Exit the loop early.
      }
    } 
    /**
     * Block Logic: If the current character is '1', increment 'countt' and reset 'counti'.
     * Checks for a dangerous situation for '1's.
     */
    else if (charr[i] == '1') {
      counti = 0;          // Reset consecutive '0's counter.
      countt = countt + 1; // Increment consecutive '1's counter.
      if (countt == 7) {   // If 7 consecutive '1's are found, it's a dangerous situation.
        break;             // Exit the loop early.
      }
    }
  }
  
  /**
   * Block Logic: Determines and prints the final result based on the counters.
   * If either counter reached 7, prints "YES"; otherwise, prints "NO".
   */
  if (counti == 7 || countt == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
