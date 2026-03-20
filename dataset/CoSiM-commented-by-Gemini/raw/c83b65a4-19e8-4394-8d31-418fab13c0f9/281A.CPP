/**
 * @file 281A.CPP
 * @brief Solution for Codeforces problem 281A, "Word Capitalization".
 * @details This program reads a word and capitalizes its first letter. If the
 * first letter is already capitalized, it remains unchanged.
 *
 * @section Algorithm
 * The algorithm for capitalizing the first letter of a string is as follows:
 * 1. Read the entire word from the standard input into a string variable.
 * 2. Iterate through the characters of the string.
 * 3. For the first character (at index 0), convert it to its uppercase
 *    equivalent and print it.
 * 4. For all subsequent characters, print them as they are without any modification.
 *
 * @subsection Complexity
 * - Time Complexity: O(N), where N is the length of the input string, because the
 *   program iterates through the string once.
 * - Space Complexity: O(N), for storing the input string.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string str;
  // Read a line of text from the standard input.
  getline(cin, str);

  /**
   * @brief This loop iterates through the input string to capitalize the first letter.
   * Pre-condition: `str` holds the word read from input.
   * Invariant: The loop processes each character once. When `i` is 0, the character
   * `str[0]` is capitalized and printed. For all other `i`, `str[i]` is printed as is.
   */
  for (int i = 0; i < str.length(); i++) {
    // Check if the current character is the first one in the string.
    if (i == 0)
      // If it is the first character, convert it to uppercase and print it.
      putchar(toupper(str[i]));
    else
      // If it's not the first character, print it without modification.
      cout << str[i];
  }

  return 0;
}
