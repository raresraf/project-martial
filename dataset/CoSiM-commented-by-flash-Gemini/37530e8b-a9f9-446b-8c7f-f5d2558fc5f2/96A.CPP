/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - Football.
 *
 * This program determines if a team is in a "dangerous" position based on a
 * sequence of 7 or more identical players (0s or 1s) in a given string.
 *
 * Algorithm: Sliding window / Linear scan. The program iterates through the
 * input string and checks for a contiguous subsequence of at least 7 identical
 * characters.
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) for storing the input string (or O(1) if input is streamed).
 */

// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h> // Required for strlen
using namespace std;

main() {
  /**
   * @brief Main function to execute the Football problem logic.
   *
   * Reads a string representing player positions (0s and 1s) and checks
   * if there is a "dangerous" situation (7 or more identical consecutive players).
   * Prints "YES" if dangerous, "NO" otherwise.
   */
  // Declare integer variables for loop counters; 'a' will store the input string.
  int n, i, j, k;
  // Declare a character array to store the input string, with a maximum size of 100.
  char a[100];
  // Read the input string into the character array 'a'.
  cin >> a;
  // Block Logic: Check if the string is long enough to contain a "dangerous" sequence.
  // Pre-condition: The string length must be at least 7 for a dangerous sequence to exist.
  if (strlen(a) > 7) {
    // Block Logic: Iterate through the string to find a sequence of 7 identical characters.
    // Invariant: 'i' represents the starting index of a potential 7-character sequence.
    // The loop runs up to 'strlen(a) - 6' to ensure there are always 7 characters
    // remaining for comparison (a[i] through a[i+6]).
    for (i = 0; i < (strlen(a) - 6); i++) {
      // Check if the current character and the next six characters are all identical.
      if (a[i] == a[i + 1] && a[i] == a[i + 2] && a[i] == a[i + 3] &&
          a[i] == a[i + 4] && a[i] == a[i + 5] && a[i] == a[i + 6]) {
        // If a dangerous sequence is found, print "YES" and terminate the program.
        cout << "YES";
        return 0;
      }
    }
  }
  // If the loop completes without finding a dangerous sequence, print "NO".
  cout << "NO";

  return 0;
}
