/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (Codeforces 118A).
 *
 * @details The problem requires processing a string with the following rules:
 * 1. Convert all uppercase letters to lowercase.
 * 2. Delete all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 3. Insert a '.' character before each consonant.
 *
 * @algorithm The program iterates through the input string character by character.
 * Inside the loop, it first converts the current character to lowercase. It then
 * checks if the character is a vowel. If it is, the loop continues to the next
 * character. If it is a consonant, the program prints a '.' followed by the
 * character.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, as it
 *                  involves a single pass.
 * Space Complexity: O(L), to store the input string.
 */
#include <ctype.h>
#include <iostream>

using namespace std;

int main() {
  char s[101];
  char aux; // aux is an unused variable.
  int i = 0;

  cin >> s;

  /**
   * @brief Block Logic: Iterates through the string to process each character.
   * @invariant `i` is the index of the character currently being processed.
   */
  while (s[i]) {
    // Convert the character to lowercase for case-insensitive vowel checking.
    s[i] = tolower(s[i]);

    // Check if the character is a vowel.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      // If it's a vowel, skip it.
      i++;
      continue;
    } else {
      // If it's a consonant, print '.' followed by the character.
      cout << "." << s[i];
      i++;
    }
  }

  cout << endl;

  return 0;
}