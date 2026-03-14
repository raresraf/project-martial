/**
 * @66687921-8125-44c2-9f42-b98bfa917620/71A.CPP
 * @brief This program solves Codeforces problem 71A, "Way Too Long Words".
 * It processes a list of words, abbreviating those longer than 10 characters.
 * Abbreviation involves replacing the middle part of the word with its length minus two,
 * e.g., "localization" (length 12) becomes "l10n". Words 10 characters or shorter
 * are printed as they are.
 *
 * Algorithm: Iterative processing of words with conditional abbreviation based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) for storing the longest word.
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process words for abbreviation.
 * Reads the number of words, then reads each word, and prints either
 * the original or abbreviated version based on its length.
 * @return 0 implicitly, as there is no explicit return statement at the end of main.
 */
int main() {
  int n; // Declares 'n', representing the number of words to process.
  cin >> n; // Reads 'n' from standard input.

  /**
   * Block Logic: Processes 'n' words. Each word is either printed as is or abbreviated based on its length.
   * Invariant: Each word from input is processed and printed according to the rules.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    string s; // Declares a string variable to store the current word.
    cin >> s; // Reads a word into 's'.

    int x = s.length(); // Gets the length of the current word.

    /**
     * Block Logic: Determines whether to print the word as is or its abbreviation.
     * Pre-condition: `x` holds the length of the current word.
     * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
     */
    if (x > 10) { // If the word length is greater than 10.
      // Prints the abbreviated form: first char, length-2, last char, followed by a newline.
      cout << s[0] << x - 2 << s[x - 1] << endl; 
    } else { // If the word length is 10 or less.
      cout << s << endl; // Prints the word as is, followed by a newline.
    }
  }
}