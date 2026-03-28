/**
 * @file 281A.CPP
 * @brief Solution to a problem that capitalizes the first letter of an input string (word or sentence).
 *
 * This program reads an entire line of text from standard input. It then iterates
 * through the characters of this string. The first character encountered is converted
 * to its uppercase equivalent, regardless of its initial case or type, and printed.
 * All subsequent characters are printed as they are. This problem is a fundamental
 * string manipulation task, often used to introduce character processing and I/O.
 *
 * Algorithm: Line-by-line input, character-by-character iteration with conditional
 *            first-character capitalization.
 * Time Complexity: O(L), where L is the length of the input string, due to string processing and I/O.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes various standard libraries (e.g., iostream, string, cctype for toupper).
#include <iostream>      // Explicitly included for std::cin, std::cout, std::getline.
#include <string.h>      // Included, but not strictly necessary as std::string is used, not C-style strings.
using namespace std;     // Using the standard namespace.

int main() {
  string str;          // Declare a string variable to store the input line.
  getline(cin, str); // Read an entire line from standard input into 'str'.

  /**
   * Block Logic: Loop through each character of the input string to apply capitalization.
   * Invariant: 'i' iterates from 0 to str.length()-1. Only the character at index 0 is modified.
   */
  for (int i = 0; i < str.length(); i++) {
    /**
     * Block Logic: Checks if the current character is the first character of the string.
     * If it is, convert to uppercase and print using putchar. Otherwise, print as is using cout.
     * Pre-condition: 'str' contains a valid string.
     */
    if (i == 0)
      // Convert the first character to uppercase using toupper() and print it using putchar().
      // toupper() is safe; it returns the character unchanged if it's not a lowercase letter.
      putchar(toupper(str[i]));
    else
      // For all other characters, print them as they are using cout.
      cout << str[i];
  }

  return 0; // Indicate successful program execution.
}
