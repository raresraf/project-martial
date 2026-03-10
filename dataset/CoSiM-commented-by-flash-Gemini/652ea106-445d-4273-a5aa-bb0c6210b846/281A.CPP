/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of an input word (string).
 *   It checks if the first character is a lowercase letter and converts it to uppercase if necessary.
 *   Problem description: "Word Capitalization", Codeforces 281A.
 * * Algorithm: Character case conversion using ASCII value manipulation.
 * Time Complexity: O(L) where L is the length of the input string, primarily due to string I/O.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.

using namespace std;

/**
 * @brief Main function to execute the string capitalization logic.
 * Functional Utility: Reads a string, capitalizes its first letter if it's lowercase, and prints the result.
 */
int main() {
  string s;
  // Block Logic: Reads the input string into 's'.
  std::cin >> s;
  /**
   * Block Logic: Checks if the first character of the string `s[0]` is a lowercase letter
   * based on its ASCII value. If it is, it converts it to its uppercase equivalent.
   *
   * Inline: `int(s[0]) > 91` is used to check for lowercase characters. ASCII values for
   *         'A'-'Z' are 65-90, and 'a'-'z' are 97-122. This condition effectively targets
   *         characters whose ASCII values fall within or above the lowercase range.
   * Inline: `s[0] = int(s[0]) - 32` converts a lowercase character to uppercase by
   *         subtracting 32, which is the consistent ASCII difference between
   *         lowercase and uppercase English letters.
   */
  if (int(s[0]) > 91)
    // s[0] = putchar(toupper(s[0])); // Alternative approach using cctype library.
    s[0] = int(s[0]) - 32;            // Converts to uppercase by adjusting ASCII value.
  // Block Logic: Prints the modified string to standard output.
  std::cout << s;
  return 0;
}