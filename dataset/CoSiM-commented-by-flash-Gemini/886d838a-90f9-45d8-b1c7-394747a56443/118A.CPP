/**
 * @file 118A.CPP
 * @brief Implements a string transformation task based on vowel and consonant rules.
 *
 * This program reads a string, converts it to lowercase, and then iterates through it.
 * It discards all vowels ('a', 'e', 'i', 'o', 'u', 'y') and, for every consonant,
 * it prints a '.' character followed by the consonant itself.
 *
 * @algorithm Single-pass string filtering.
 * @time_complexity O(N), where N is the length of the input string, as it iterates through the string once.
 * @space_complexity O(1), as the transformation is printed directly to standard output without using additional data structures proportional to the input size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int i;
  char s[100];
  // Functional Utility: Reads a word from standard input into a character array.
  scanf("%s", s);
  // cout<<s<<endl;
  int x = strlen(s);

  /**
   * @brief Block Logic: Iterates through each character of the input string for processing.
   *
   * The loop processes the string from left to right, applying vowel/consonant
   * rules to each character.
   * @invariant `i` is the index of the character currently being processed.
   */
  for (i = 0; i < x; i++) {
    // Functional Utility: Converts the current character to its lowercase equivalent to ensure case-insensitive vowel checking.
    s[i] = tolower(s[i]);

    /**
     * @brief Block Logic: Filters characters based on whether they are vowels.
     *
     * This conditional block checks if the character is one of the defined vowels.
     * @pre-condition The character `s[i]` is in lowercase.
     * @invariant If `s[i]` is a vowel, the loop continues to the next iteration, effectively ignoring it. Otherwise, the `else` block is executed.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      continue;
    } else {
      // Block Logic: Handles consonant characters.
      // A '.' is prepended to the consonant before printing to standard output.
      cout << "." << s[i];
    }
  }
  // cout<<s;
}