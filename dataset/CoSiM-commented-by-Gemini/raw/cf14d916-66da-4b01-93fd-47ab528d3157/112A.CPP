/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A, "Petya and Strings".
 * @details This program reads two strings and compares them lexicographically,
 *          ignoring differences in case. It prints -1 if the first string is
 *          lexicographically smaller, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Iterate through both strings, converting any uppercase characters to lowercase.
 *    This is done by adding 32 to the ASCII value of the uppercase character.
 * 3. Use the standard C++ `std::string` comparison operators (`==`, `<`, `>`)
 *    on the now-lowercase strings to determine their lexicographical order.
 * 4. Print the corresponding result (0, -1, or 1).
 *
 * @complexity
 * Time: O(L), where L is the length of the strings. The conversion loop and the
 *       final comparison both take linear time.
 * Space: O(L) to store the two input strings.
 */
#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
int main() {
  string word;
  string word2;

  cin >> word;
  cin >> word2;

  /**
   * @brief Convert both strings to lowercase to ensure a case-insensitive comparison.
   * @note This loop uses manual ASCII value manipulation (`+ 32`) instead of the
   *       standard `tolower()` function.
   */
  for (int s = 0; s < (int)word.length(); s++) {
    if (isupper(word[s]))
      word[s] = word[s] + 32;
    else
      ; // Empty else statement has no effect.
    if (isupper(word2[s]))
      word2[s] = word2[s] + 32;
    else
      ; // Empty else statement has no effect.
  }

  /**
   * @brief Perform the lexicographical comparison on the modified lowercase
   *        strings and print the required output.
   */
  if (word == word2)
    cout << "0" << endl;

  else if (word < word2)
    cout << "-1" << endl;

  else if (word > word2)
    cout << "1" << endl;

  return 0;
}
