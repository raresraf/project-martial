/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 * @details This program processes a string according to a specific set of rules:
 *          1. It removes all vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts).
 *          2. Before each consonant, it inserts a '.' character.
 *          3. It converts all uppercase consonants to their lowercase equivalents.
 *
 * @algorithm
 * The program iterates through the input string character by character. For each character, it checks
 * if it is a vowel by searching for it in a predefined string of vowels.
 * - If the character is a vowel, it is skipped.
 * - If the character is a consonant, a '.' is printed, followed by the lowercase version of the consonant.
 *   Lowercase conversion is done by checking if the character is uppercase and adding the ASCII difference if so.
 *
 * @complexity
 * - Time: O(L), where L is the length of the input string. The `find` operation on the short, fixed-size vowel string takes constant time.
 * - Space: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s; // The input string to be processed.
  // A string containing all vowels (both cases) for easy lookup. 'Y' is treated as a vowel in this problem.
  string f = "aoyeuiAOYEUI";
  cin >> s;

  int i = -1;
  /**
   * @brief Main loop to iterate through each character of the input string.
   */
  while (++i < s.size()) {
    // Pre-condition: Check if the current character s[i] is a vowel.
    if (f.find(s[i]) != string::npos)
      // Action: The character is a vowel, so skip to the next character.
      continue;
    else
      /**
       * Action: The character is a consonant. Print a '.' and then the consonant converted to lowercase.
       * The ternary operator handles the conversion: if the character's ASCII value is greater
       * than 'Z', it's assumed to be already lowercase; otherwise, it's converted.
       */
      cout << "." << (char)(s[i] > 'Z' ? s[i] : s[i] + ('a' - 'A'));
  }
}