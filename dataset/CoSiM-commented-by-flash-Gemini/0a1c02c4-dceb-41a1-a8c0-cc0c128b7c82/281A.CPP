/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - Word Capitalization.
 *
 * This problem asks to capitalize the first letter of a given word (string) and print the result.
 *
 * Algorithm: Character manipulation.
 * The code reads a single word (string), converts its first character to uppercase using `toupper()`,
 * and then prints the modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string, due to string input/output
 * and a constant number of operations on the first character.
 * Space Complexity: O(L) for storing the input string.
 */

//
// Created by Taewoo Kim on 3/23/2017.
//

#include <iostream> // For standard C++ input/output operations (cin, cout).
#include <string> // For using the `string` class.
#include <cctype> // For character manipulation functions like `toupper`.

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Word Capitalization problem.
 * Functional Utility: Reads a string, capitalizes its first character, and then prints the modified string.
 * Algorithm: Directly modifies the first character of the input string in place.
 */
int main() {
  string s; // Declares a string variable `s` to store the input word.
  cin >> s; // Reads the input word into string `s`.
  
  // Functional Utility: Converts the first character of the string `s` to its uppercase equivalent.
  // Precondition: `s` is not an empty string.
  // Postcondition: If `s[0]` is a lowercase letter, it becomes uppercase; otherwise, it remains unchanged.
  s[0] = toupper(s[0]);
  
  cout << s; // Prints the modified string `s` to standard output.
  return 0; // Indicates successful program execution.
}
