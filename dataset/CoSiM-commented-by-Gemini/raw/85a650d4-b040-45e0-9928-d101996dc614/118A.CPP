/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from competitive programming.
 *
 * @details
 * The program processes an input string based on a set of rules:
 * 1. It removes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. It inserts a '.' character before every consonant.
 * 3. It converts all uppercase consonants to their lowercase equivalents.
 *
 * The program reads the input character by character and builds the output string
 * on the fly.
 *
 * Time Complexity: O(N), where N is the length of the input string, as each character is processed once.
 * Space Complexity: O(1), as it processes the string as a stream without storing it all in memory.
 */
#include <ctype.h>
#include <stdio.h>

/**
 * @brief Checks if a character is a vowel.
 * Vowels are defined as 'A', 'O', 'Y', 'E', 'U', 'I' (case-insensitive).
 * @param c The character to check.
 * @return True (1) if the character is a vowel, false (0) otherwise.
 */
bool isVowel(char c) {
  c = tolower(c);
  if (c == 'e' || c == 'a' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {
    return 1;
  }
  return 0;
}

int main() {
  char temp;
  /**
   * @brief Reads the input character by character and processes it.
   * Loop continues until the end of the input stream.
   */
  while (scanf("%c", &temp) == 1) {
    /**
     * @brief Process the character if it is a consonant.
     * Whitespace characters are ignored.
     */
    if (!isVowel(temp) && !isspace(temp)) {
      // If it's a consonant, print a '.' followed by its lowercase version.
      printf(".%c", tolower(temp));
    }
  }
  printf("\n");
  return 0;
}
