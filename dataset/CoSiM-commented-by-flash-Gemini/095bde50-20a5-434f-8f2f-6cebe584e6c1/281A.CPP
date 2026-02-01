/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of an input string.
 * It reads a line of text from standard input, converts its first character to uppercase,
 * and then prints the modified string to standard output.
 * Algorithm: String manipulation.
 * Time Complexity: O(L) where L is the length of the input string, due to string operations.
 * Space Complexity: O(L) to store the input string.
 */

#include <iostream> // Required for input/output operations (cin, cout, getline).
#include <string>   // Required for string manipulation.
#include <cctype>   // Required for character manipulation functions like toupper.

using namespace std; // Uses the standard namespace to avoid prefixing standard library elements with std::.

int main() {
  string line; // Declares a string variable to store the input line of text.
  // Functional Utility: Reads an entire line of text from standard input into the 'line' variable.
  getline(cin, line);

  // Block Logic: Capitalizes the first character of the input string.
  // Pre-condition: The 'line' string is not empty.
  // Invariant: Only the character at index 0 of the string is modified.
  // Post-condition: The first character of 'line' is converted to its uppercase equivalent.
  line[0] = toupper(line[0]);

  // Functional Utility: Prints the modified string to standard output, followed by a newline.
  cout << line << endl;
  return 0; // Indicates successful program execution.
}