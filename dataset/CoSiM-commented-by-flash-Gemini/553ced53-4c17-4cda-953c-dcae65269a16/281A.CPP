/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given input string if it is a lowercase letter.
 *
 * This program reads a string and checks if its first character is a lowercase alphabet.
 * If it is, the character is converted to its uppercase equivalent. Finally, the
 * (potentially modified) string is printed to the console.
 *
 * Algorithm: Character case conversion based on ASCII values.
 * Time Complexity: O(1) - Accessing and modifying the first character is a constant time operation.
 * Space Complexity: O(N) - where N is the length of the input string, to store the string.
 */

#include <iostream>

/**
 * @brief Uses the standard namespace for easier access to common C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return 0 if the program executes successfully.
 */
int main() {
  /**
   * Block Logic: Declares a character array to store the input string.
   * Invariant: 's' will contain the input string read from standard input.
   */
  char s[1000];
  
  /**
   * Block Logic: Reads the input string from standard input into the character array 's'.
   */
  cin >> s;
  
  /**
   * Block Logic: Checks if the first character of the string is a lowercase letter.
   * Pre-condition: 's' contains a null-terminated string.
   * Invariant: If the first character is lowercase, it will be converted to uppercase.
   */
  if (s[0] >= 'a' && s[0] <= 'z') {
    /**
     * Inline: Converts the lowercase first character to uppercase by subtracting 32 from its ASCII value.
     * This relies on the ASCII values of 'a'-'z' and 'A'-'Z' being consecutively offset by 32.
     */
    s[0] = s[0] - 32;
  }
  
  /**
   * Block Logic: Prints the modified string to standard output.
   */
  cout << s;
  
  /**
   * Functional Utility: Indicates successful program termination.
   */
  return 0;
}
