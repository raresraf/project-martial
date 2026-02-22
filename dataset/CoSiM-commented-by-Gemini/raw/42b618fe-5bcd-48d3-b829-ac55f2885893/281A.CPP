/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A - Word Capitalization.
 *
 * @algorithm
 * The problem requires capitalizing the first letter of a given word.
 * The rest of the word should remain unchanged.
 *
 * The algorithm is as follows:
 * 1. Read the input word into a character array `s`.
 * 2. Access the first character of the array, `s[0]`.
 * 3. Convert this character to its uppercase equivalent using `toupper()` and
 *    store it back into `s[0]`. This modification is done in-place.
 * 4. Iterate through the modified character array and print each character to
 *    output the final capitalized word.
 *
 * @note This solution uses a C-style fixed-size character array and functions
 *       like `strlen` instead of the more modern C++ `std::string`.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the word. This is dominated
 *   by reading the input and then printing the output character by character
 *   (which requires a `strlen` call, also O(L)).
 * - Space Complexity: O(L) to store the word in the character array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {

  char s[1003];
  cin >> s;
  // Capitalize the first letter of the word in-place.
  s[0] = toupper(s[0]);
  /**
   * @brief Print the modified word character by character.
   * `strlen(s)` is called to determine the length of the C-style string.
   */
  for (int i = 0; i < strlen(s); i++)
    cout << s[i];
  return 0;
}
