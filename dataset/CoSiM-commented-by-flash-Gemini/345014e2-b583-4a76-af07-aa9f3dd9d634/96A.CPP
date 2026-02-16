/**
 * @file 96A.CPP
 * @brief Solution for a competitive programming problem that determines if a given string contains seven consecutive identical characters.
 *
 * Algorithm: Linear scan with consecutive character count.
 * Time Complexity: O(L) where L is the length of the input string. The algorithm iterates through the string once.
 * Space Complexity: O(1) as only a few constant-size variables are used for counting and flags.
 */

#include <cstring>
#include <iostream>
using namespace std;

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Reads a string from standard input and determines if it
 * contains any sequence of seven or more identical consecutive characters.
 * It prints "YES" if such a sequence is found, and "NO" otherwise.
 *
 * @return 0 on successful execution.
 */
int main() {
  string s; // @brief Input string to be analyzed.
  int flag = 0; // @brief Flag to indicate if a dangerous (seven identical consecutive characters) sequence is found (1 for found, 0 for not found).
  cin >> s;
  int count = 1; // @brief Counter for consecutive identical characters.
  // @brief Loop variable for iterating through the string.
  for (int i = 1; i < s.length(); i++) {
    // Block Logic: Check if the current character is identical to the previous one.
    // Invariant: `count` stores the number of consecutive identical characters ending at `s[i]`.
    if (s[i - 1] == s[i]) {
      count++;
    } else {
      // Functional Utility: Reset the counter if the characters are different.
      count = 1;
    }
    // Block Logic: Check if seven consecutive identical characters have been found.
    // Invariant: If `count` reaches 7, a "dangerous" sequence is detected.
    if (count == 7) {
      cout << "YES"; // Functional Utility: Output "YES" indicating a dangerous sequence was found.
      flag = 1;
      break;
    }
  }
  // Block Logic: Check the `flag` to determine the final output.
  // Invariant: If `flag` is still 0, no dangerous sequence was found throughout the string.
  if (flag == 0) {
    cout << "NO"; // Functional Utility: Output "NO" indicating no dangerous sequence was found.
  }
  return 0;
}
