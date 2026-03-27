/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string.
 *        It robustly handles the capitalization by first converting the
 *        character to lowercase (if needed) and then to uppercase.
 *
 * Algorithm:
 * 1. Read a single word (string) from standard input.
 * 2. Convert the first character of the string to lowercase (if it isn't already),
 *    then convert it to uppercase using arithmetic manipulation based on ASCII values.
 * 3. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string length check and character access.
 * Space Complexity: O(L) to store the input string.
 */

#include "bits/stdc++.h"
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 on successful execution.
 */
int main(int argc, char const *argv[]) {
  string str;
  // Block Logic: Reads a single word (string) from standard input.
  cin >> str;

  /**
   * Block Logic: Converts the first character of the string to uppercase.
   * This is done by first ensuring it's treated as a lowercase character
   * (via `tolower`) and then converting it to uppercase by adjusting its
   * ASCII value relative to 'a' and 'A'.
   *
   * Pre-condition: The string 'str' has been populated and is not empty.
   * Invariant: The first character of 'str' will be an uppercase letter.
   */
  str[0] = (tolower(str[0]) - 'a' + 'A');
  // Block Logic: Prints the modified string to standard output.
  cout << str;
  return 0;
}
