/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 * @details This program reads a single word and capitalizes its first letter.
 * If the first letter is already capitalized or is not a letter, it remains unchanged.
 *
 * @algorithm
 * 1. Reads a word from standard input into a string.
 * 2. Accesses the first character of the string using index 0.
 * 3. Uses the `toupper()` function from the `<cctype>` library to convert this character
 *    to its uppercase equivalent.
 * 4. The original string, now modified in-place, is printed to standard output.
 *
 * @complexity
 * - Time: O(L) where L is the length of the word, dominated by I/O operations. The capitalization is an O(1) operation.
 * - Space: O(L) to store the word in a string.
 */
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string inp; // String to store the input word.
  
  /**
   * @brief Reads words from input until EOF.
   * For this problem's constraints, it will typically execute only once.
   */
  while (cin >> inp) {
    // Access the first character and convert it to uppercase.
    // The `toupper` function handles characters that are already uppercase or not letters gracefully.
    inp[0] = toupper(inp[0]);
    
    // Print the modified string with the capitalized first letter.
    cout << inp << endl;
  }
}