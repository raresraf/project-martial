/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A, "Word Capitalization".
 *
 * Algorithm:
 * This program capitalizes the first letter of a given word. It reads a word
 * from standard input, checks if the first character is a lowercase letter
 * (from 'a' to 'z'), and if so, converts it to its uppercase equivalent using
 * the `toupper()` function. If the first letter is already capitalized or is
 * not a letter, the word is printed unchanged.
 *
 * The code is placed within a `while` loop, allowing it to process multiple
 * words from the input stream, although the original problem only requires
 * processing one.
 *
 * Time Complexity: O(N) for a single word of length N, dominated by string I/O and copy operations.
 * Space Complexity: O(N) for a single word, as it stores two copies of the input string.
 */
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  // s: original input string.
  // s1: a copy of the input string to be modified.
  string s, s1;
  int i, j;
  
  /**
   * @brief A loop to process words from standard input until EOF.
   * For the specific problem, this loop will only run once.
   */
  while (cin >> s) {
    s1 = s; // Redundant copy of the string.
    
    /**
     * @brief Checks if the first character is a lowercase letter.
     */
    if (s1[0] >= 'a' && s1[0] <= 'z') {
      // If it is lowercase, convert it to uppercase.
      s1[0] = toupper(s1[0]);
      cout << s1 << endl;
    } else {
      // Otherwise, print the original string.
      cout << s << endl;
    }
  }
  return 0;
}
