/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" competitive programming problem.
 * @details This program reads a single word and capitalizes its first letter, leaving
 * the rest of the word unchanged.
 *
 * @section algorithm Algorithm
 * The algorithm is straightforward:
 * 1. Read the input word into a standard C++ string `s`.
 * 2. Access the first character of the string using index 0 (`s[0]`).
 * 3. Use the `toupper()` function from the cctype library to convert this character
 *    to its uppercase equivalent. The original character is replaced in the string.
 * 4. Print the modified string to standard output.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L) where L is the length of the string, dominated by the input/output operations. The capitalization itself is an O(1) operation.
 * - Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>  // For using the string class
#include <cctype>  // For using the toupper function

using namespace std;

int main() {
  string s;
  // Read the word from standard input.
  cin >> s;

  // Check if the string is not empty before capitalizing.
  if (!s.empty()) {
    // Capitalize the first letter of the string.
    // The toupper() function converts a character to its uppercase version.
    // The result is assigned back to the first character of the string.
    s[0] = toupper(s[0]);
  }

  // Print the modified string.
  cout << s;

  return 0;
}
