/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string.
 *        It uses `islower` to check if the first character is lowercase and
 *        `toupper` to convert it to uppercase.
 *
 * Algorithm:
 * 1. Include necessary headers for input/output and character manipulation.
 * 2. Read a single word (string) from standard input.
 * 3. Check if the first character of the string is a lowercase letter using `islower`.
 * 4. If it is lowercase, convert the first character to uppercase using `toupper`.
 * 5. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string operations.
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>
#include <cctype>
#include <iostream>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @return 0 on successful execution.
 */
int main() {
  string str;
  // Block Logic: Reads a single word (string) from standard input.
  cin >> str;
  /**
   * Block Logic: Conditionally converts the first character of the string to uppercase.
   * Uses `islower` to determine if the character is a lowercase letter, and `toupper`
   * to perform the conversion. This is a robust method that handles various character sets correctly.
   *
   * Pre-condition: The string 'str' has been populated and is not empty.
   * Invariant: If the first character was lowercase, it will be uppercase; otherwise, it remains unchanged.
   */
  if (islower(str[0])) {
    // Inline: Converts the character to its uppercase equivalent using a standard library function.
    str[0] = toupper(str[0]);
  }
  // Block Logic: Prints the potentially modified string to standard output.
  cout << str;
  return 0;
}
