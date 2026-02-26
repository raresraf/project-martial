/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given string.
 *
 * This program reads a single string from standard input and ensures that its
 * first character is uppercase. If the first character is already uppercase
 * or not a letter, it remains unchanged.
 *
 * @algorithm Direct character manipulation.
 * @complexity
 * Time Complexity: O(1) - The program only inspects and potentially modifies
 * the first character of the string, regardless of the string's length.
 * Space Complexity: O(N) - Where N is the length of the input string, to
 * store the string itself.
 */
#include <iostream>
#include <string> // It's good practice to include the <string> header for std::string.

// Using the standard namespace to avoid prefixing std:: to every standard library call.
using namespace std;

// Global string variable to hold the input.
string a;

int main() {
  // Read the input string from the user.
  cin >> a;

  /**
   * @block
   * @brief Ensures the first character is uppercase.
   * @logic Checks if the ASCII value of the first character `a[0]` corresponds
   * to a lowercase letter (ASCII > 96). If it is, it converts it to its
   * uppercase equivalent by subtracting 32 from its ASCII value.
   */
  if ((int)a[0] > 96) {
    a[0] -= 32;
  }

  // Print the modified or original string to standard output.
  cout << a;

  return 0; // Though not strictly necessary in modern C++, it's good practice to return 0 from main.
}
