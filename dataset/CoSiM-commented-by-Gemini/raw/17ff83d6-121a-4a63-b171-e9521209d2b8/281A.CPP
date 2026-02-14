/**
 * @file 281A.CPP
 * @brief Solution for a problem that requires capitalizing the first letter of a string.
 * @details This program reads a single word and capitalizes its first character, then prints the modified word.
 *
 * @section algorithm Algorithm
 * 1. Read the input word into a string variable `s`.
 * 2. Access the first character of the string using index `0` (i.e., `s[0]`).
 * 3. Use the `toupper()` function from the `<cctype>` library (implicitly included via `<iostream>` in some compilers)
 *    to convert this character to its uppercase equivalent. The result is assigned back to `s[0]`.
 * 4. Print the modified string to the standard output.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the operation is on the first character only, regardless of the string's length.
 *   (Note: Reading and printing the string takes O(L) where L is the length, but the core logic is O(1)).
 * - Space Complexity: O(L), to store the input string.
 */
#include <iostream>
#include <cctype> // For toupper()

using namespace std;

/**
 * @brief Main function to capitalize the first letter of a word.
 * @return 0 on successful execution.
 */
int main() {
  string s;
  // Read the input word.
  cin >> s;
  
  // Capitalize the first character of the string.
  s[0] = toupper(s[0]);
  
  // Print the resulting string.
  cout << s;
  
  return 0;
}
