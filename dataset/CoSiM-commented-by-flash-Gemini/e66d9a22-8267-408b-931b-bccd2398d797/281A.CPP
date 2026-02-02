/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem: Word Capitalization.
 *
 * Algorithm: String manipulation (capitalize first letter).
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L), where L is the length of the input string.
 *
 * This program reads a single word and capitalizes its first letter, then prints the modified word.
 */

#include <iostream>
#include <locale> // Required for std::toupper
// Functional Utility: Simplifies standard namespace access for competitive programming.
using namespace std;

int main() {
  string str;
  // Functional Utility: Reads a single word (string) from standard input.
  cin >> str;

  // Functional Utility: Converts the first character of the string to uppercase.
  str[0] = toupper(str[0]);

  // Functional Utility: Prints the modified string to standard output, followed by a newline.
  cout << str << endl;

  return 0;
}
