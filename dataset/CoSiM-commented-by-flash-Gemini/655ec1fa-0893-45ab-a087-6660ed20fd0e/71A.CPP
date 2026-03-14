/**
 * @655ec1fa-0893-45ab-a087-6660ed20fd0e/71A.CPP
 * @brief This program solves Codeforces problem 71A, "Way Too Long Words".
 * It processes a list of words, abbreviating those longer than 10 characters.
 * Abbreviation involves replacing the middle part of the word with its length minus two,
 * e.g., "localization" (length 12) becomes "l10n". Words 10 characters or shorter
 * are printed as they are.
 *
 * Algorithm: Iterative processing of words with conditional abbreviation based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(N * L_max) for storing all words in a vector of strings.
 */
// Codeforces
// 71A - Way too long word

#include <iostream> // For standard input/output operations (cin, cout).
#include <vector>   // For using std::vector.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process words for abbreviation.
 * Reads the number of words, then reads each word, storing them in a vector.
 * Finally, iterates through the vector to print either the original or abbreviated version.
 * @return 0 on successful execution.
 */
int main() {
  int size; // Declares 'size', representing the number of words to process.
  cin >> size; // Reads 'size' from standard input.

  vector<string> vs; // Declares a vector of strings to store the input words.
  string line;       // Declares a string variable to temporarily store each word read.

  /**
   * Block Logic: Reads 'size' words from standard input and stores them in the `vs` vector.
   * Invariant: After this loop, `vs` contains all the input words.
   */
  for (int i = 0; i < size; i++) { // Loops `size` times, once for each word.
    cin >> line;                   // Reads a word into `line`.
    vs.push_back(line);            // Adds the word to the vector.
  }

  /**
   * Block Logic: Iterates through the stored words and applies the abbreviation rule.
   * Invariant: Each word is processed and printed according to the rules: abbreviated if
   *            longer than 10 characters, otherwise printed as is.
   */
  for (int i = 0; i < size; i++) { // Loops through each word in the vector.
    /**
     * Block Logic: Determines whether to print the word as is or its abbreviation.
     * Pre-condition: `vs[i].size()` holds the length of the current word.
     * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
     */
    if (vs[i].size() <= 10) // If the word length is 10 or less.
      cout << vs[i] << endl; // Prints the word as is, followed by a newline.
    else { // If the word length is greater than 10.
      cout << vs[i][0]; // Prints the first character.
      cout << (vs[i].size() - 2); // Prints the length minus 2.
      cout << vs[i][vs[i].size() - 1] << endl; // Prints the last character, followed by a newline.
    }
  }

  return 0; // Indicates successful program termination.
}
