/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details The program abbreviates words that are longer than 10 characters.
 * An abbreviation consists of the first character of the word, followed by
 * the number of characters between the first and last character, and then the
 * last character. Words with 10 or fewer characters are not modified.
 *
 * Algorithm:
 * 1. Read the number of words to process, 'n'.
 * 2. Loop 'n' times:
 *    a. Read a word into a character array.
 *    b. Calculate the length of the word.
 *    c. If the length is greater than 10, print the abbreviation:
 *       (first char) + (length - 2) + (last char).
 *    d. Otherwise, print the word as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word, dominated by reading the input strings.
 * Space Complexity: O(L), for the buffer to store the longest word.
 */
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n;         // The number of words to process.
  int len;       // To store the length of the current word.
  char str[102]; // A C-style string (character array) to store the input word.

  // Read the total number of words.
  cin >> n;

  /**
   * Block Logic: Process each of the 'n' words.
   * This loop iterates 'n' times, handling one word per iteration.
   */
  while (n--) {
    cin >> str; // Read the word.
    len = strlen(str); // Calculate its length.

    // Check if the word is "too long".
    if (len <= 10) {
      // If the word has 10 or fewer characters, print it unchanged.
      cout << str << endl;
    } else {
      // If the word is longer than 10 characters, print its abbreviation.
      // The abbreviation is the first character, the count of middle characters (len-2), and the last character.
      cout << str[0] << len - 2 << str[len - 1] << endl;
    }
  }
  return 0;
}