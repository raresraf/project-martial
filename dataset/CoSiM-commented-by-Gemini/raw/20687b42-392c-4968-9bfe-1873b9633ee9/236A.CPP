/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A, which determines a user's gender based on the parity of distinct characters in their username.
 * @details This program reads a username and counts the number of unique characters. If the count of unique characters is even, it outputs "CHAT WITH HER!"; otherwise, it outputs "IGNORE HIM!".
 *
 * @algorithm Frequency Counting
 * @complexity
 * - Time: O(N), where N is the length of the input string, for iterating through the string once.
 * - Space: O(1), as the boolean array `abc` has a fixed size of 26, corresponding to the English alphabet.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string name;
  bool abc[26];
  int diffLet = 0;

  /**
   * @brief Initializes a boolean array to track the presence of each letter of the alphabet.
   * @invariant All elements of `abc` are `false`.
   */
  for (int i = 0; i < 26; i++)
    abc[i] = false;

  cin >> name;

  /**
   * @brief Iterates through the input string to mark the presence of each character.
   * @details For each character in the string, the corresponding index in the `abc` array is set to `true`.
   *          This effectively builds a set of unique characters present in the name.
   */
  for (int i = 0; i < name.length(); i++) {
    abc[name[i] - 'a'] = true;
  }

  /**
   * @brief Counts the number of distinct characters found in the string.
   * @details It iterates through the boolean array and increments a counter for each `true` value.
   */
  for (int i = 0; i < 26; i++) {
    if (abc[i])
      diffLet++;
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * @details If the number of different letters (`diffLet`) is even, the program prints "CHAT WITH HER!".
   *          Otherwise (if it's odd), it prints "IGNORE HIM!".
   */
  cout << (diffLet % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!") << endl;
}