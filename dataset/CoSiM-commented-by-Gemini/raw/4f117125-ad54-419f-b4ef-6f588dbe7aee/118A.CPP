/**
 * @file 118A.CPP
 * @brief A C-style solution to the "String Task" problem from Codeforces.
 *
 * @details The program processes a string according to three rules:
 * 1. Convert all uppercase letters to their lowercase equivalents.
 * 2. Delete all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 3. Before each consonant, insert a '.' character.
 * This version uses C-style strings and I/O functions.
 *
 * @algorithm
 * The program iterates through the input string in a single pass.
 * 1. For each character, it first converts it to lowercase if applicable.
 * 2. It then checks if the character is a vowel.
 * 3. If it is a vowel, the `continue` statement skips the rest of the loop
 *    body and proceeds to the next character.
 * 4. If the character is not a vowel (a consonant), the code proceeds to print
 *    a '.' followed by the consonant character.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, as it involves
 * a single pass.
 * Space Complexity: O(N) to store the input C-style string.
 */
#include <stdio.h>
#include <string.h>

int main() {
  char ch[105]; // A C-style string to hold the input.
  int len, i;

  scanf("%s", ch);

  // Calculate the length once before the loop for efficiency.
  len = strlen(ch);

  /**
   * @block
   * @brief Iterate through the string to process each character.
   */
  for (i = 0; i < len; i++) {
    // Convert uppercase letters to lowercase.
    if (ch[i] >= 'A' && ch[i] <= 'Z')
      ch[i] = 'a' + (ch[i] - 'A'); // Portable conversion.

    // If the character is a vowel, skip to the next iteration.
    if (ch[i] == 'a' || ch[i] == 'e' || ch[i] == 'i' || ch[i] == 'o' ||
        ch[i] == 'u' || ch[i] == 'y') {
      continue;
    } else {
      // If it's not a vowel (i.e., a consonant), print it with a preceding dot.
      printf(".%c", ch[i]);
    }
  }
  printf("\n");

  return 0;
}