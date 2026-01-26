/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem from Codeforces.
 *
 * This problem requires processing a list of words. For each word, if its
 * length is strictly greater than 10 characters, it should be abbreviated
 * to its first character, followed by the number of characters between the
 * first and last character, and then its last character. Otherwise, the
 * word should be printed as is.
 *
 * Problem Link: https://codeforces.com/problemset/problem/71/A
 *
 * Algorithm: Iterative string processing with conditional abbreviation.
 * Time Complexity: O(N * L), where N is the number of words and L is the
 *                  average length of a word. String length calculation and
 *                  character access are typically O(1) for C++ strings.
 * Space Complexity: O(L) to store the current word.
 */

#include <iostream>
#include <string> // Required for string manipulation

using namespace std;

int main() {
  int n; // Declares an integer to store the number of words to process.
  // Reads the number of words 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Processes each word based on its length for abbreviation or direct output.
   * Invariant: Each iteration handles one word from the input, applying the abbreviation rule.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    string s;  // Declares a string to store the current word.
    // Reads the current word 's' from standard input.
    cin >> s;

    int si = s.length(); // Calculates the length of the current word.

    // Conditional abbreviation: if the word is too long (length > 10).
    if (si > 10) {
      // Prints the abbreviated form: first character, (length - 2), last character.
      cout << s[0] << si - 2 << s[si - 1] << '\n';
    } else {
      // Otherwise, prints the word as is.
      cout << s << '\n';
    }
  }

  return 0;
}