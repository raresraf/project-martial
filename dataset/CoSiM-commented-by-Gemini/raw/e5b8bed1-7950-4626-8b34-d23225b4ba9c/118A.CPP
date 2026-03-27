/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details This program processes a string according to specific rules:
 *          1. It converts all uppercase letters to lowercase.
 *          2. It removes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *          3. It inserts a character "." before each consonant.
 *
 * @section
 * Algorithm:
 * The program iterates through the input string character by character. For each
 * character, it first converts it to lowercase. It then checks if the character
 * is a vowel. If it is, the character is ignored. If it is a consonant, a '.'
 * is printed, followed by the consonant itself.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(N), where N is the length of the input string, as it involves a single pass through the string.
 * - Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  // Pre-condition: Reads one or more strings from standard input until EOF.
  while (cin >> s) {
    // Block Logic: Iterates through each character of the string to apply the transformation rules.
    for (int i = 0; i < s.size(); i++) {
      // Converts the character to lowercase for case-insensitive vowel checking.
      s[i] = tolower(s[i]);
      // Block Logic: Filters out vowels. Consonants are processed in the else block.
      // Invariant: Vowels are identified and skipped.
      if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
          s[i] == 'u' || s[i] == 'y') {

      } else
        // Action: For any consonant, prepend a '.' and print it.
        cout << '.' << s[i];
    }
    // Post-condition: A newline is printed after processing each input string.
    cout << endl;
  }
  return 0;
}
