/**
 * @5a048587-2b91-4dea-b11e-7455303391a3/71A.CPP
 * @brief This program solves problem 71A, known as "Way Too Long Words".
 * It takes an integer 'n' followed by 'n' words. For each word, if its length
 * exceeds 10 characters, it is abbreviated to its first character, followed by
 * its length minus 2, and then its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 *
 * Algorithm: Iterative processing of words with conditional abbreviation based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum length of a word.
 *                  (Due to `strlen` and string printing for each word).
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string.h> // For strlen (C-style string length function).
using namespace std; // Uses the standard namespace.

int main() {
  int n, i;      // Declares 'n' for the number of words, 'i' for word length.
  char str[1000]; // Declares a C-style character array to store input words (max 999 chars + null terminator).

  cin >> n; // Reads the number of words 'n' from standard input.

  /**
   * Block Logic: Processes 'n' words. Each word is either printed as is or abbreviated based on its length.
   * Invariant: Each word from input is processed and printed according to the rules.
   */
  if (n >= 1) { // Ensures there is at least one word to process.
    while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
      cin >> str; // Reads a word into 'str'.
      i = strlen(str); // Calculates the length of the read word.
      
      /**
       * Block Logic: Determines whether to print the word as is or its abbreviation.
       * Pre-condition: 'i' holds the length of 'str'.
       * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
       */
      if ((i >= 1) && (i <= 10)) // If the word length is between 1 and 10 (inclusive).
        cout << str << endl; // Prints the word as is, followed by a newline.
      else if (i <= 100) // If the word length is greater than 10 (and <= 100, though this upper bound is not critical here).
        // Prints the abbreviated form: first char, length-2, last char.
        cout << str[0] << (i - 2) << str[i - 1] << endl; 
    }
  }

  return 0; // Indicates successful program termination.
}
