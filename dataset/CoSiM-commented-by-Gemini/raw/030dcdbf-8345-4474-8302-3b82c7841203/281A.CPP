/**
 * @file 281A.CPP
 * @brief Solution for Codeforces problem 281A "Word Capitalization".
 *
 * This program takes a single word as input and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * Algorithm: Read the word, convert the first character to uppercase, and then
 * print the modified word.
 *
 * Time Complexity: O(L) where L is the length of the input word, due to string operations.
 * Space Complexity: O(L) for storing the word.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  // Block Logic: Reads a single word from standard input.
  cin >> word;
  /**
   * Block Logic: Capitalizes the first character of the word and prints the result.
   * 1. `static_cast<char>(toupper(word[0]))`: Converts the first character to uppercase.
   * 2. `word.substr(1, word.size())`: Extracts the rest of the string from the second character onwards.
   * 3. The two parts are concatenated and printed to standard output, followed by a newline.
   * Pre-condition: `word` is a non-empty string.
   */
  cout << static_cast<char>(toupper(word[0])) << word.substr(1, word.size())
       << endl;
  return 0;
}
