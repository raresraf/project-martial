/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * This program transforms an input string by converting uppercase letters to lowercase,
 * deleting vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a "." before each consonant.
 *
 * Algorithm: Character-by-character processing. Each character is checked for being a vowel
 * or a consonant, converted to lowercase, and then printed with a preceding dot if it's a consonant.
 * Time Complexity: O(L), where L is the length of the input string, as each character is processed once.
 * Space Complexity: O(1) beyond input/output storage, as processing is done in-place per character.
 */

#include <ctype.h> // @brief Required for character handling functions like tolower() and isspace().
#include <stdio.h> // @brief Required for C-style input/output operations like scanf() and printf().

bool isVowel(char c) {
  c = tolower(c);
  if (c == 'e' || c == 'a' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {
    return 1;
  }
  return 0;
}

int main() {
  /**
   * @brief Main function to perform the string transformation task.
   * Reads characters one by one from standard input, applies the transformation rules
   * (lowercase, remove vowels, add '.' before consonants), and prints the result.
   */
  char temp; // @brief Temporary character variable to store each character read from input.
  
  // Block Logic: Read characters from standard input until EOF.
  // Invariant: Each character is processed according to the rules before the next is read.
  while (scanf("%c", &temp) == 1) { // Inline: scanf("%c", &temp) returns 1 on successful read.
    // Block Logic: Apply transformation rules.
    // Check if the character is not a vowel and not a whitespace character.
    if (!isVowel(temp) && !isspace(temp)) {
      // Inline: If it's a consonant, print a '.' followed by its lowercase version.
      printf(".%c", tolower(temp));
    }
  }
  printf("\n"); // @brief Print a newline character at the end of the output.
  return 0; // @brief Indicates successful program termination.
}
