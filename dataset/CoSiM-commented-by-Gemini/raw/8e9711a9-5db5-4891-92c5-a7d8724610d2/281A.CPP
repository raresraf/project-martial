/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem (e.g., Codeforces 281A).
 * @details This program reads a single word and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * @algorithm Direct in-place string modification.
 * @complexity
 * - Time: O(L) where L is the length of the word, dominated by the I/O operation
 *   to read the string. The capitalization of the first character is an O(1) operation.
 * - Space: O(L) to store the input word.
 */
#include <iostream>
#include <string>   // Include for std::string
#include <cctype>   // Include for toupper

using namespace std;

int main() {
  string word;

  // Read the word from standard input.
  cin >> word;
  
  // Access the first character (at index 0) and convert it to uppercase.
  // The result of toupper is cast back to a char and assigned back,
  // modifying the string in-place.
  word[0] = char(toupper(word[0]));
  
  // Print the modified string.
  cout << word;

  return 0;
}
