#include <iostream>
#include <string>

using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @details This program shortens any word longer than 10 characters into a special
 * abbreviation. Shorter words are printed unchanged.
 *
 * Algorithm:
 * The program first reads an integer specifying the number of words to process.
 * It then loops that many times. In each iteration, it reads a word and checks its
 * length. If the length is greater than 10, it constructs and prints an abbreviation:
 * the first character, followed by the number of characters between the first and last
 * (i.e., length - 2), and then the last character. If the word's length is 10 or less,
 * the word is printed as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is their average length,
 * due to the I/O operations in the loop.
 * Space Complexity: O(L_max), where L_max is the maximum length of a word, to store it.
 */
int main() {
  string word;
  int num;
  cin >> num;
  /**
   * @brief Loop to process each word.
   */
  for (int i = 0; i < num; i++) {
    cin >> word;
    // Check if the word is "too long".
    if (word.length() > 10) {
      // Print the abbreviated version.
      cout << word.at(0) << word.length() - 2 << word.at(word.length() - 1)
           << endl;
    } else {
      // Print the word as it is.
      cout << word << endl;
    }
  }
}
