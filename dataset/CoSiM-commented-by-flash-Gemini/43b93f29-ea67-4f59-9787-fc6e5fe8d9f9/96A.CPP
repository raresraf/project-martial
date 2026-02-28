/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 96A - "Football").
 *
 * This program determines if a "dangerous situation" occurs in a football game.
 * A dangerous situation is defined as having 7 or more consecutive players of the
 * same team on the field. The players are represented by '0's and '1's in a string.
 *
 * Algorithm: String traversal and consecutive character counting.
 * The program iterates through the input string, maintaining a count of consecutive
 * '0's and '1's. If either count reaches 7, a flag is set, and the process terminates early.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * The string is traversed at most once.
 * Space Complexity: O(1), as only a few integer variables are used in addition to input string storage.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a binary string, checks for 7 consecutive identical characters, and prints "YES" or "NO".
 */
int main() {
  int len;           // Variable to store the length of the input string.
  int zero = 0;      // Counter for consecutive '0's.
  int one = 0;       // Counter for consecutive '1's.
  int flag = 0;      // Flag to indicate if a dangerous situation is found (1 = YES, 0 = NO).
  string s;          // Variable to store the input string.

  // Pre-condition: Read the input string.
  cin >> s;
  len = s.length(); // Get the length of the string.

  // Block Logic: Iterate through each character of the string.
  // Invariant: 'i' represents the current index in the string.
  for (int i = 0; i < len; i++) {
    // Block Logic: If the current character is '0'.
    if (s[i] == '0') {
      zero++; // Increment consecutive '0's count.
      one = 0; // Reset consecutive '1's count.
    } else { // Block Logic: If the current character is '1'.
      one++;  // Increment consecutive '1's count.
      zero = 0; // Reset consecutive '0's count.
    }
    // Block Logic: Check if either consecutive count has reached 7.
    // Functional Utility: This is an early exit condition.
    if (zero == 7 || one == 7) {
      flag = 1;  // Set flag to indicate a dangerous situation.
      break;     // Exit the loop early.
    }
  }
  // Block Logic: Print the result based on the 'flag'.
  if (flag == 1)
    cout << "YES" << endl; // Dangerous situation found.
  else
    cout << "NO" << endl;  // No dangerous situation.
  return 0; // Indicate successful execution.
}
