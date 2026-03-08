/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of an input string if it is lowercase.
 *
 * Algorithm: Direct character manipulation based on ASCII values.
 * Time Complexity: O(1) due to constant time operations.
 * Space Complexity: O(N) for storing the input string, where N is the length of the string.
 */
#include <iostream>
using namespace std;

// Global variable to store the input string.
string a;

/**
 * @brief Main function to read a string and capitalize its first letter.
 * @pre None.
 * @post The modified string (with the first letter capitalized if applicable) is printed to standard output.
 */
int main() {
  // Functional Utility: Reads the input string from standard input.
  cin >> a;

  /**
   * Block Logic: Checks if the first character of the string is a lowercase letter.
   * Pre-condition: 'a[0]' contains the first character of the input string.
   * Invariant: If 'a[0]' is between 'a' (ASCII 97) and 'z' (ASCII 122), it is converted to uppercase.
   *            Subtracting 32 from its ASCII value converts a lowercase English letter to its uppercase equivalent.
   */
  if ((int)a[0] > 96) { // ASCII value for 'a' is 97. Checks if the character is lowercase.
    a[0] -= 32; // Converts the lowercase character to uppercase.
  }
  // Functional Utility: Prints the potentially modified string to standard output.
  cout << a;
  // Indicates successful program execution.
  return 0;
}