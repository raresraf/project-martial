/**
 * @644e9d69-42d9-4f2e-bf79-b91ef3bd7117/71A.CPP
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
#include <cstdio>   // For C-style input/output functions, not explicitly used here.
#include <cstring>  // For C-style string functions, not explicitly used here.
#include <fstream>  // For file stream operations, not explicitly used here.
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process words for abbreviation.
 * Reads words one by one and prints either the original or abbreviated version.
 * @return 0 on successful execution.
 */
int main() {
  int n;     // Declares 'n', representing the number of words to process.
  cin >> n;  // Reads 'n' from standard input.
  string s;  // Declares a string variable to store the current word.

  /**
   * Block Logic: Processes 'n' words. Each word is either printed as is or abbreviated based on its length.
   * Invariant: Each word from input is processed and printed according to the rules.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each word.
    cin >> s; // Reads a word into 's'.

    /**
     * Block Logic: Determines whether to print the word as is or its abbreviation.
     * Pre-condition: `s.size()` holds the length of the current word.
     * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
     */
    if (s.size() > 10) // If the word length is greater than 10.
      // Prints the abbreviated form: first char, length-2, last char, followed by a newline.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << "\n"; 
    else // If the word length is 10 or less.
      cout << s << "\n"; // Prints the word as is, followed by a newline.
  }
  return (0); // Indicates successful program termination.
}