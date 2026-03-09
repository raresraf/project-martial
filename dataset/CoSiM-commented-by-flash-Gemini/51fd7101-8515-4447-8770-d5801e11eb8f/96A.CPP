/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: "Football".
 *        This program determines if a given football situation (represented as a string of '0's and '1's)
 *        is "dangerous". A situation is dangerous if there are 7 or more consecutive players of the same team.
 *
 * Algorithm: Iterative string traversal with a counter for consecutive identical characters.
 * Time Complexity: O(L) where L is the length of the input string, due to a single pass iteration.
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Manages the main logic flow to read the football situation string,
   *                     check for dangerous patterns, and print the result.
   */
  string s;          // Declares a string to store the input football situation.
  getline(cin, s);   // Reads the entire line (the football situation string) from standard input.
  int len = s.size();  // Stores the length of the input string.
  int c = 1;         // Counter for consecutive identical characters, initialized to 1.
  bool f = false;    // Flag to indicate if a dangerous situation is found, initialized to false.
  /**
   * Block Logic: Iterates through the string to find consecutive identical characters.
   * Pre-condition: `s` contains the input string; `len` is its length; `c` is the consecutive counter;
   *                `f` is the dangerous situation flag.
   * Invariant: `c` holds the current count of consecutive identical characters; `f` is set to true
   *            if a sequence of 7 or more is found. The loop terminates early if `f` becomes true.
   */
  for (int i = 1; i < len; i++) {
    if (s[i] == s[i - 1]) { // If the current character is the same as the previous one.
      c++;                  // Increment the consecutive counter.
      if (c == 7) {         // If 7 consecutive identical characters are found.
        f = true;           // Set the dangerous flag to true.
        break;              // Exit the loop early as a dangerous situation is confirmed.
      }
    } else {              // If the current character is different from the previous one.
      c = 1;              // Reset the consecutive counter to 1 for the new character.
      f = false;          // Reset the dangerous flag, as the consecutive sequence is broken.
    }
  }
  printf("%s\n", f ? "YES" : "NO"); // Prints "YES" if a dangerous situation was found, "NO" otherwise.

  return 0; // Indicate successful program termination.
}
