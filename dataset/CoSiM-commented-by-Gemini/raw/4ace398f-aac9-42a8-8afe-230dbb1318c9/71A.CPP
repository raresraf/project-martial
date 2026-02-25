/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A (Way Too Long Words).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program reads a list of words and shortens any word that is longer
 * than 10 characters. Shorter words are printed as they are.
 *
 * @section problem_description
 * The abbreviation format is the first letter of the word, followed by the
 * number of letters between the first and last characters, and then the last
 * letter of the word. For example, "localization" becomes "l10n".
 *
 * @section algorithm
 * 1. Read an integer `n`, which is the total number of words to process.
 * 2. Enter a loop that repeats `n` times.
 * 3. In each iteration, read a word into a string.
 * 4. Check if the string's length is greater than 10.
 * 5. If it is, print the abbreviated form: first character, length minus two,
 *    and the last character.
 * 6. If not, print the original word.
 *
 * @section complexity
 * - Time Complexity: O(N * L), where N is the number of words and L is the
 *   maximum length of a word, dominated by I/O operations.
 * - Space Complexity: O(L) to store the longest word.
 */

#include <iostream>
#include <string> // Recommended to explicitly include for std::string
using namespace std;

int main() {
  // n: The number of test cases (words).
  int n;
  cin >> n;

  string s; // To store each word.

  /**
   * @brief Main loop to process each word.
   *
   * The loop continues until all `n` words have been processed.
   */
  while (n--) {
    cin >> s;

    // Check if the length of the word exceeds 10 characters.
    if (s.length() > 10) {
      // If the word is "too long", print its abbreviation.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // Otherwise, print the word as is.
      cout << s << endl;
    }
  }

  return 0;
}
