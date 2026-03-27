// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string.
 *        It uses the `toupper` function, which converts lowercase letters to
 *        uppercase and leaves other characters unchanged.
 *
 * Algorithm:
 * 1. Read a single word (string) from standard input.
 * 2. Apply the `toupper` function to the first character of the string.
 * 3. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string length check and character access.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @return 0 on successful execution.
 */
int main() {
  string s;
  // Block Logic: Reads a single word (string) from standard input.
  cin >> s;
  /**
   * Block Logic: Converts the first character of the string to its uppercase equivalent.
   * Functional Utility: Ensures the first character is capitalized, regardless of its initial case.
   */
  s[0] = toupper(s[0]);
  // Block Logic: Prints the modified string to standard output, followed by a newline.
  cout << s << endl;
  return 0;
}