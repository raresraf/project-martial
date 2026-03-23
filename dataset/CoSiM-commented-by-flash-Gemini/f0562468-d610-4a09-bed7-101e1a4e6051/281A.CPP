/**
 * @file 281A.CPP
 * @brief This competitive programming solution takes a single word as input
 *        and capitalizes its first letter if it is currently in lowercase.
 *        If the first letter is already uppercase, it remains unchanged.
 * Algorithm: Conditional Character Manipulation.
 *            1. Reads a string representing a word.
 *            2. Checks the ASCII value of the first character to determine if it's lowercase.
 *            3. If lowercase, it converts the character to uppercase by subtracting 32 from its ASCII value.
 *            4. Prints the modified word.
 * Time Complexity: O(L), where L is the length of the input word. This includes
 *                  reading the word and a constant-time check/modification of the first character.
 * Space Complexity: O(L), for storing the input word string.
 */
#include <iostream> // Required for standard input/output operations (cin, cout, endl).
#include <string>   // Required for string manipulation.
// #include <string.h> // Included but not strictly necessary for std::string operations.
using namespace std;

/**
 * @brief Main function of the program. Reads a word, capitalizes its first letter if needed, and prints it.
 * Functional Utility: Serves as the entry point for the "Word Capitalization" problem.
 */
int main() {
  string word;   // Declares a string variable to hold the input word.
  cin >> word;   // Reads the word from standard input.

  // Block Logic: Checks if the first character of the word is a lowercase letter.
  // Invariant: The condition `int(word[0]) >= 97` effectively checks if the character's ASCII value
  //            falls within the range of 'a' through 'z'.
  if (int(word[0]) >= 97) { // ASCII value for 'a' is 97.
    // Converts the first character to uppercase by subtracting 32 from its ASCII value.
    // (e.g., 'a' (97) - 32 = 'A' (65)).
    word[0] = char(int(word[0]) - 32);
  }
  cout << word << endl; // Prints the modified word followed by a newline.
  return 0;             // Indicates successful program termination.
}
