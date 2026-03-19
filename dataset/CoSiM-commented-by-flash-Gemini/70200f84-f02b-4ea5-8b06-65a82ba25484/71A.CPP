/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words".
 * Abbreviates words that are longer than 10 characters according to a specific rule:
 * the first character, followed by the count of characters between the first and last,
 * followed by the last character. Words 10 characters or shorter remain unchanged.
 * Algorithm: Read each word, check its length, and apply the abbreviation rule if necessary.
 * Time Complexity: O(N * L_max), where N is the number of words and L_max is the maximum length of a word.
 * Space Complexity: O(L_max) to store the longest word.
 */
// the-Quert 71A - Way Too Long Words // Original problem reference comment.
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for string manipulation.

using namespace std;

/**
 * @brief Main function to solve the "Way Too Long Words" problem.
 * Reads a number of words and processes each one for abbreviation based on its length.
 */
int main() {
  int n;      // Number of words to be processed.
  string s;   // String to store each word read from input.
  cin >> n;   // Reads the number of words (test cases).

  /**
   * Block Logic: Processes each word one by one.
   * Invariant: After each iteration, one word has been read, processed, and its result printed.
   * The loop continues until all 'n' words have been handled.
   */
  while (n--) {
    cin >> s; // Reads the current word.
    /**
     * Block Logic: Checks if the word's length exceeds 10 characters.
     */
    if (s.length() > 10) {
      // Abbreviate the word: print first character, then (length - 2), then last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // If the word is 10 characters or shorter, print it as is.
      cout << s << endl;
    }
  }
  return 0;
}
