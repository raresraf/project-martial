/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (281A) from Codeforces.
 * @details This program capitalizes the first letter of a given word if it's lowercase.
 * This implementation relies on direct manipulation of ASCII character codes.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The program checks the ASCII value of the first character to determine if
 * it's a lowercase letter and then converts it if necessary.
 * 1. Read the input word.
 * 2. Check if the ASCII value of the first character is greater than 96. In the ASCII
 *    table, all lowercase English letters have values from 97 ('a') to 122 ('z').
 * 3. If it is, subtract 32 from its ASCII value to convert it to its uppercase equivalent
 *    (e.g., 'a' (97) - 32 = 'A' (65)).
 * 4. Print the modified word.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the word, dominated by the I/O operations.
 * The core logic is O(1).
 * Space Complexity: O(L), to store the input word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string word;
  cin >> word;
  /**
   * @block
   * @brief Capitalizes the first letter if it is lowercase, using ASCII arithmetic.
   */
  // Checks if the first character's ASCII value is in the lowercase range (>96).
  if (word[0] > 96) {
    // Converts to uppercase by subtracting the ASCII difference (32).
    word[0] = word[0] - 32;
  }
  cout << word;

  return 0;
}
