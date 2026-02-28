/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem from Codeforces.
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first character, the number of middle
 * characters, and the last character. Shorter words are printed unchanged.
 *
 * @algorithm String Length Check and Direct Output
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times to process each word.
 * 3. For each word, get its length.
 * 4. If the length is greater than 10, print the first character, the length
 *    minus two, and the last character.
 * 5. Otherwise, print the word as is.
 *
 * @complexity
 * - Time: O(N * L_max), where N is the number of words and L_max is the maximum
 *   length of a word, dominated by reading the input strings.
 * - Space: O(L_max), to store the longest word.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of words to process.
  int n;
  cin >> n;

  /**
   * @brief Process each of the `n` words.
   * `while (n--)` is a common idiom in competitive programming for a loop
   * that runs `n` times.
   */
  while (n--) {
    // `input`: stores the current word.
    string input;
    cin >> input;
    int length = input.length();

    // Pre-condition: Check if the word's length exceeds 10 characters.
    if (length > 10) {
      // If it's too long, print the abbreviation in the format:
      // [first letter][number of middle letters][last letter]
      cout << input[0] << length - 2 << input[length - 1] << endl;
    } else {
      // If the word is not too long, print it unchanged.
      cout << input << endl;
    }
  }

  return 0; // Indicate successful execution.
}
