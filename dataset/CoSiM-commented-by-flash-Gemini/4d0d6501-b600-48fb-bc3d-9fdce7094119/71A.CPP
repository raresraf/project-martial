/**
 * @file 71A.CPP
 * @brief Implements a word abbreviation logic: words longer than 10 characters are abbreviated (e.g., "localization" becomes "l10n").
 *
 * Algorithm: Iterative word processing. Checks word length and applies abbreviation or direct printing.
 * Time Complexity: O(N * L_avg), where N is the number of words and L_avg is the average length of a word.
 *                  Operations on `std::string` (size, access) are typically O(1).
 * Space Complexity: O(L_max) for storing the longest word.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for std::string manipulations.

using namespace std;

/**
 * @brief Main function to process a list of words according to abbreviation rules.
 * @pre Input consists of an integer 'n' (number of words), followed by 'n' words.
 * @post For each word, prints its abbreviation or the word itself if its length is 10 or less.
 */
int main() {
  int n;      // Variable to store the number of words to process.
  // Functional Utility: Reads the count of words from standard input.
  cin >> n;

  string s; // Variable to store the current word being processed.

  /**
   * Block Logic: Processes each word 'n' times.
   * Invariant: In each iteration, a new word is read, checked for length, and then either abbreviated or printed as is.
   */
  while (n--) { // Loop 'n' times.

    // Functional Utility: Reads the current word from standard input.
    cin >> s;

    /**
     * Block Logic: Applies abbreviation if the word's length exceeds 10 characters.
     * Pre-condition: 's' holds the current word.
     * Invariant: If 's.size()' is greater than 10, the word is abbreviated to its first character,
     *            the count of intermediate characters, and its last character. Otherwise, it's printed as is.
     */
    if (s.size() > 10) { // Checks if the word length is greater than 10.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl; // Prints abbreviated form.
    } else
      cout << s << endl; // Prints the word as is.
  }

  // Indicates successful program execution.
  return 0;
}