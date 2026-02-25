/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - Stones on the Table.
 *
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones are of the same color. It achieves this
 * by counting adjacent identical stones.
 *
 * Algorithm: Linear scan. Iterate through the array of stones and increment a
 * counter whenever two adjacent stones have the same color.
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) for storing the input string.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to execute the Stones on the Table problem logic.
   *
   * Handles optional file redirection for local testing, reads the number
   * of stones and their colors, calculates the number of identical adjacent
   * stones, and prints the result.
   */

// Block Logic: Conditional compilation for local debugging with file I/O redirection.
// Invariant: This block is only active when not compiling for online judges (ONLINE_JUDGE is not defined).
#ifndef ONLINE_JUDGE
  // Redirects standard input from "input.txt" for easier local testing.
  freopen("input.txt", "r", stdin);
  // Redirects standard output to "output.txt" for easier local testing.
  freopen("output.txt", "w", stdout);
#endif

  // Declare 'n' for the total number of stones and 'a' to count adjacent identical stones.
  int n = 0, a = 0;
  // Read the total number of stones.
  cin >> n;
  // Declare 's' to store the string representing the colors of the stones.
  string s;
  // Read the string of stone colors.
  cin >> s;
  // Block Logic: Iterate through the stones to find adjacent identical colors.
  // Invariant: 'i' iterates from the first stone up to the second-to-last stone,
  // comparing each stone with its immediate successor.
  for (int i = 0; i < n - 1; i++) {
    // If the current stone and the next stone are the same color, increment the counter 'a'.
    if (s[i] == s[i + 1]) {
      a += 1;
    }
  }
  // Output the total count of adjacent identical stones.
  cout << a << endl;

  return 0;
}
