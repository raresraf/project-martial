/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A (Way Too Long Words).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program processes a series of words. If a word's length is more than
 * 10 characters, it is shortened into a specific abbreviation. Otherwise, it is
 * printed as is.
 *
 * @section problem_description
 * The abbreviation is formed by taking the first letter of the word, the number
 * of letters between the first and last character, and the last letter of the
 * word. For instance, "localization" is abbreviated to "l10n".
 *
 * @section algorithm
 * 1. An integer `n` is read, representing the number of words to follow.
 * 2. The program loops `n` times.
 * 3. Inside the loop, a word is read.
 * 4. Its length is checked. If it's greater than 10, the program prints the
 *    abbreviation.
 * 5. Otherwise, the original word is printed.
 *
 * This implementation uses `string::front()` and `string::back()` (C++11 features)
 * for a slightly more expressive way to access the first and last characters.
 *
 * @section complexity
 * - Time Complexity: O(N * L), where N is the number of words and L is the
 *   maximum word length, dominated by string input operations.
 * - Space Complexity: O(L) to store the longest word.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  // n: the number of words to process.
  int n;
  cin >> n;

  string input; // To store each word.

  /**
   * @brief Main loop to process each of the n words.
   * The `n-- > 0` is a concise way to loop n times.
   */
  while (n-- > 0) {
    cin >> input;

    // Check if the word is "too long".
    if (input.length() > 10) {
      // Print the abbreviation using front() and back() for clarity.
      cout << input.front() << input.length() - 2 << input.back() << endl;
    } else {
      // Print the word unchanged if it's not too long.
      cout << input << endl;
    }
  }
  return 0;
}
