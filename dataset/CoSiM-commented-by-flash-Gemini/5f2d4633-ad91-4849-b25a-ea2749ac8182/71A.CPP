/**
 * @5f2d4633-ad91-4849-b25a-ea2749ac8182/71A.CPP
 * @brief This program solves problem 71A, "Way Too Long Words".
 * It reads an integer 'n' followed by 'n' words. For each word, if its length
 * exceeds 10 characters, it is abbreviated to its first character, followed by
 * its length minus 2, and then its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 *
 * Algorithm: Iterative processing of words with conditional abbreviation based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.

using namespace std; // Uses the standard namespace.

int main() {
  int n; // Declares 'n', representing the number of words to process.
  cin >> n; // Reads 'n' from standard input.

  /**
   * Block Logic: Processes 'n' words. Each word is either printed as is or abbreviated based on its length.
   * Invariant: Each word from input is processed and printed according to the rules.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    string input; // Declares a string variable to store the current word.
    cin >> input; // Reads a word into 'input'.
    int length = input.length(); // Gets the length of the read word.

    /**
     * Block Logic: Determines whether to print the word as is or its abbreviation.
     * Pre-condition: 'length' holds the length of 'input'.
     * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
     */
    if (length > 10) // If the word length is greater than 10.
      // Prints the abbreviated form: first char, length-2, last char, followed by a newline.
      cout << input[0] << length - 2 << input[length - 1] << endl; 
    else // If the word length is 10 or less.
      cout << input << endl; // Prints the word as is, followed by a newline.
  }

  return 0; // Indicates successful program termination.
}