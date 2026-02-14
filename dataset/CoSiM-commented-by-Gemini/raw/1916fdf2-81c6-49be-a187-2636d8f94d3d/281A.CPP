/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem.
 * @details This program reads a single word and capitalizes its first letter if it is
 * a lowercase letter. It then prints the modified word.
 *
 * @section algorithm Algorithm
 * 1. Read the input word into a string variable `a`.
 * 2. Check the ASCII value of the first character, `a[0]`. If the value is greater
 *    than 96, it signifies that the character is a lowercase letter (since 'a' is 97).
 * 3. If the character is indeed lowercase, convert it to uppercase by subtracting 32
 *    from its ASCII value.
 * 4. Print the modified string to the standard output.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L) where L is the length of the string, dominated by the I/O operations.
 *   The core capitalization logic is O(1).
 * - Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

string a; // Input string is declared globally.

/**
 * @brief Main function to capitalize the first letter of a word.
 * @return 0 on successful execution.
 */
int main() {
  // Read the input word.
  cin >> a;
  
  // Check if the first character is a lowercase letter by its ASCII value.
  if ((int)a[0] > 96) {
    // Convert to uppercase by subtracting 32 from the ASCII value.
    a[0] -= 32;
  }
  
  // Print the resulting string.
  cout << a;
  
  return 0;
}