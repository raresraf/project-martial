/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem using a 2D character array.
 *
 * This program abbreviates words longer than 10 characters. It first reads all
 * words into a 2D array of characters and then processes them in a separate loop.
 * This implementation uses C-style strings and the `strlen` function.
 *
 * Algorithm:
 * 1. Read the number of test cases, n.
 * 2. Read all 'n' words and store them in a 2D character array.
 * 3. Loop 'n' times through the stored words:
 *    a. Calculate the length of the current word.
 *    b. If the length is greater than 10, print its abbreviation.
 *    c. Otherwise, print the original word.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * maximum word length.
 * Space Complexity: O(T * L) to store all words in the 2D array.
 */
#include <iostream>
#include <string.h> // For strlen

using namespace std;

int main() {

  int n, len;
  // A 2D array to store up to 100 words, each with a max length of 100.
  char str[100][101];

  cin >> n; // Read the number of words.

  /**
   * Block Logic: Input Phase
   * This loop reads all 'n' words from standard input and stores them.
   */
  for (int i = 0; i < n; ++i) {
    cin >> str[i];
  }

  /**
   * Block Logic: Processing and Output Phase
   * This loop iterates through the stored words to apply the abbreviation logic.
   */
  for (int i = 0; i < n; ++i) {

    // 'len' is calculated as the number of characters *between* the first and last.
    // This is an unconventional way to check the length.
    len = strlen(str[i]) - 2;

    // The condition `len <= 8` is equivalent to `strlen(str[i]) <= 10`.
    if (len <= 8) {
      // If the word's total length is 10 or less, print it as is.
      cout << str[i] << endl;

    } else {
      // If the word is too long, print the abbreviation.
      // str[i][0]: The first character.
      // len: The count of middle characters.
      // str[i][len + 1]: The last character (index is strlen - 2 + 1 = strlen - 1).
      cout << str[i][0] << len << str[i][len + 1] << endl;
    }
  }

  return 0;
}
