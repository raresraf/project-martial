/**
 * @5eda4357-eec3-4634-aa3b-73f57ca6f45b/281A.CPP
 * @brief This program solves the "Word Capitalization" problem (Codeforces 281A).
 * It reads a single word, capitalizes its first letter, and then prints the modified word.
 * It includes an unused helper function `conver`.
 *
 * Algorithm: Direct string manipulation - specifically, in-place modification of the first character.
 * Time Complexity: O(L) where L is the length of the input word. Effectively O(1) for typical word lengths.
 * Space Complexity: O(L) for storing the string. Effectively O(1) for typical word lengths.
 */

#include <algorithm> // Included but not explicitly used in active main.
#include <cmath>     // Included but not explicitly used in active main.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <string>    // For using std::string.
#include <vector>    // Included but not explicitly used in active main.
#include <cctype>    // For std::toupper.

using namespace std; // Uses the standard namespace.

/**
 * @brief Converts a character representing a digit to its integer value.
 * This function is declared but not used in the `main` function.
 * @param a The character digit (e.g., '0', '1').
 * @return The integer value of the digit.
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
int conver(char a) { 
  return a - 48; // Inline: Converts char to int by subtracting ASCII value of '0'.
}

int main() {
  string a; // Declares a string variable to store the input word.
  cin >> a; // Reads the word from standard input.
  
  // Block Logic: Converts the first letter of the input string to uppercase.
  // Pre-condition: The string 'a' is not empty.
  // Invariant: The first character of 'a' is converted to its uppercase equivalent.
  // Inline: `toupper()` converts a character to its uppercase form if it's a lowercase letter;
  //         otherwise, it returns the character unchanged.
  a[0] = toupper(a[0]); 
  cout << a; // Prints the modified string.
  // The program implicitly returns 0 on successful completion of main.
}