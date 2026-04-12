/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 *
 * This program attempts to determine if a username (string) represents a "boy" or a "girl"
 * based on the parity of the number of distinct characters in the username.
 * If the count of distinct characters is odd, the output should be "IGNORE HIM!".
 * If the count is even, the output should be "CHAT WITH HER!".
 *
 * Algorithm (Intended): Attempt to count distinct characters using nested loops and a flag.
 * Time Complexity: O(L^2) where L is the length of the string, due to nested loops.
 * Space Complexity: O(1) beyond the input string.
 * Note: The implemented logic for counting distinct characters may not be entirely accurate
 * for all edge cases due to how duplicates are handled.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to classify a username as "boy" or "girl".
 *
 * Functional Utility: Reads a username string, processes it to count characters
 * based on a specific (and potentially flawed) duplicate-checking mechanism,
 * and prints a message based on the parity of this count.
 */
int main() {
  string s1;
  cin >> s1; // Read the username string from standard input.
  int k = 0, l = 0; // 'k' intended to count distinct characters, 'l' is a duplicate flag.

  /**
   * Block Logic (Intended): Iterates through the string to identify and count characters
   * that are considered "distinct" by the implemented logic.
   * The inner loop checks for duplicates of s1[i] further down the string.
   * If a duplicate is found, 'l' is set, and s1[i] is effectively skipped from being counted by 'k'.
   * Pre-condition: 's1' contains the input username string.
   * Invariant (Intended): 'k' accumulates a count related to unique character occurrences.
   */
  for (int i = 0; i < s1.length(); i++) {
    for (int j = i + 1; j < s1.length(); j++)
      if (s1[i] == s1[j])
        l = 1; // Mark 'l' if a duplicate of s1[i] is found later in the string.

    if (l == 1) { // If s1[i] had duplicates after it.
      l = 0;      // Reset duplicate flag for next iteration.
      continue;   // Skip counting this character in 'k'.
    }
    k++; // Increment 'k' if s1[i] was not found to have duplicates *after* its current position.
  }
  /**
   * Block Logic: Determines the output message based on the parity of 'k'.
   * Prints "CHAT WITH HER!" if 'k' is even, and "IGNORE HIM!" if 'k' is odd.
   * Post-condition: One of two predefined messages is printed to standard output.
   */
  if (k % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}