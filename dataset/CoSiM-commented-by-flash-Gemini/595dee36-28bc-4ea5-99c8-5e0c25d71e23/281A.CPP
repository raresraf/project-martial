/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (Problem 281A).
 *        This program reads a single word, capitalizes its first letter if it's
 *        originally lowercase, and then prints the modified word.
 * Algorithm: String Manipulation / Character Case Conversion
 * Time Complexity: O(L) for reading the input string, O(1) for capitalizing the first character,
 *                  where L is the length of the input word.
 * Space Complexity: O(L) to store the input string.
 */

#include <cstring>  // Potentially included for C-style string functions, though not directly used for std::string.
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for using std::string.

using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 * @return 0 if the program executes successfully.
 */
int main() {
  string word; // Declare a string to store the input word.

  // Read the input word from standard input.
  cin >> word;

  /**
   * Block Logic: Checks if the first character of the word is a lowercase letter.
   * Pre-condition: 'word' contains the input string.
   * Post-condition: If the first character is lowercase, it is converted to uppercase.
   */
  if (word[0] >= 'a' && word[0] <= 'z') {
    // Inline: Convert the first character to uppercase by subtracting 32 from its ASCII value.
    // This is a common trick, as 'A' (65) is 32 less than 'a' (97) in ASCII.
    word[0] = word[0] - 32;
  }

  // Print the modified word followed by a newline.
  cout << word << endl;

  return 0; // Indicate successful program termination.
}
