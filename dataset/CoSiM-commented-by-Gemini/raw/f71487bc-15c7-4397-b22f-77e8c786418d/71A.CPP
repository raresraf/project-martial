/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (71A).
 * @details This program reads a number of words and for each word, it checks if its length is
 * strictly greater than 10. If it is, the word is replaced by a special abbreviation,
 * otherwise it remains unchanged. The abbreviation consists of the first character of the word,
 * followed by the number of characters between the first and last character, and finally the last character.
 *
 * Algorithm:
 * 1. Read the number of words to process.
 * 2. Loop for that many times:
 *    a. Read a word.
 *    b. Check if its length is greater than 10.
 *    c. If it is, construct the abbreviated string.
 *    d. Print the original or abbreviated word.
 *
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum length of a word,
 * due to reading the input strings. The processing of each word is O(1) assuming string conversions are constant time for reasonable lengths.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

int main(void) {
  int lineCount;
  string line;

  // Pre-condition: lineCount is the number of words to process.
  cin >> lineCount;
  /**
   * Block Logic: Iterates through each word, applying the abbreviation logic if necessary.
   * Invariant: At the start of each iteration, `i` words have been processed and printed.
   */
  for (int i = 0; i < lineCount; i++) {
    cin >> line;
    size_t length = line.length();
    // Pre-condition: Check if the word length exceeds the threshold for abbreviation.
    if (length > 10) {
      // Logic: The word is abbreviated to its first character, the count of
      // intermediate characters, and its last character.
      line = line[0] + to_string(length - 2) + line[length - 1];
    }
    cout << line << endl;
  }

  return 0;
}