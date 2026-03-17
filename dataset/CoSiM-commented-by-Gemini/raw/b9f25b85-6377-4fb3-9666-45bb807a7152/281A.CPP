/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 *
 * @details This program reads a single word and capitalizes its first letter if it is not already uppercase.
 * It then prints the resulting word to the standard output.
 *
 * @section
 * Algorithm:
 * 1. Read the input word.
 * 2. Check if the first character is a lowercase letter.
 * 3. If it is, convert it to its uppercase equivalent.
 * 4. Print the modified word.
 *
 * Time Complexity: O(L), where L is the length of the word, dominated by the string input operation.
 * Space Complexity: O(L), to store the word.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string a;
  cin >> a;

  /**
   * @brief This block checks if the first character of the string is a lowercase letter.
   * Pre-condition: The string 'a' must not be empty.
   * Logic: If the character's ASCII value is not in the range of 'A' to 'Z', it's treated as
   * a lowercase letter that needs to be capitalized.
   */
  if (!('A' <= a[0] && a[0] <= 'Z'))
    // Inline: Converts the character to uppercase using ASCII arithmetic.
    // The difference 'A' - 'a' gives the constant offset between uppercase and lowercase letters.
    a[0] += 'A' - 'a';

  cout << a << endl;
  return 0;
}
