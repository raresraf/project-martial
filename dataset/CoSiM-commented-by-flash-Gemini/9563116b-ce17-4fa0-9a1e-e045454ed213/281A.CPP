/**
 * @file 281A.CPP
 * @brief Solution to a problem that capitalizes the first letter of a given word.
 *
 * This program reads a single word (string) as input. It then checks if the first
 * character of the word is a lowercase letter. If it is, the program converts
 * this first character to its uppercase equivalent by subtracting 32 from its
 * ASCII value. Finally, it prints the modified word to standard output.
 * This is a common string manipulation task often encountered in programming challenges.
 *
 * Algorithm: Direct string character access, conditional ASCII value adjustment.
 * Time Complexity: O(L), where L is the length of the input string (dominated by I/O operations).
 *                  The core capitalization logic operates in O(1) time.
 * Space Complexity: O(L) for storing the input string.
 */
#include <algorithm> // Generally included for std::transform, but not used here.
#include <iostream>  // Required for std::cin, std::cout.
#include <string>    // Required for std::string.
#include <vector>    // Not used in this specific implementation.

using namespace std; // Using the standard namespace to simplify code.

int main() {

  string text; // Declare a string variable to store the input word.

  // Read a string (word) from standard input.
  cin >> text;

  /**
   * Block Logic: Checks if the first character of the input word is a lowercase letter.
   * If true, it converts the first character to its uppercase equivalent.
   * Pre-condition: 'text' contains at least one character.
   */
  if (text[0] >= 'a' && text[0] <= 'z') { // Check if the first character is between 'a' and 'z' (inclusive).
    // Convert the first character to uppercase by subtracting 32 from its ASCII value.
    // (e.g., 'a' (97) - 32 = 'A' (65)).
    text[0] -= 32;
  }

  // Print the modified string to standard output, followed by a newline.
  cout << text << endl;

  return 0; // Indicate successful program execution.
}
