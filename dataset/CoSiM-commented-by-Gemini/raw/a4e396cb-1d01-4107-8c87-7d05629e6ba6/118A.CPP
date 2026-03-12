/**
 * @file 118A.CPP
 * @brief C-style solution to the "String Task" problem from Codeforces.
 *
 * This program processes a string according to a set of rules: it removes all
 * vowels, converts all consonants to lowercase, and prepends a '.' to each
 * resulting consonant.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the length of the input string, as it iterates through the string once.
 * @space_complexity O(N) to store the input string in a character array.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, i;
  char inp[105]; // Buffer to hold the input string.

  scanf("%s", inp);
  n = strlen(inp);

  /**
   * @brief Iterates through the input string to process each character.
   *
   * The loop checks each character:
   * 1. If it's a vowel (case-insensitive), it is skipped.
   * 2. If it's a consonant, it is converted to lowercase (if not already)
   *    and printed with a preceding '.'.
   */
  for (i = 0; i < n; i++) {
    // Check for any vowel, upper or lower case.
    if ((inp[i] == 'A') || (inp[i] == 'a') || (inp[i] == 'E') ||
        (inp[i] == 'e') || (inp[i] == 'I') || (inp[i] == 'i') ||
        (inp[i] == 'O') || (inp[i] == 'o') || (inp[i] == 'U') ||
        (inp[i] == 'u') || (inp[i] == 'Y') || (inp[i] == 'y'))
      continue; // Skip vowels.
    else if (inp[i] > 'a') // A simple check if the consonant is already lowercase.
      printf(".%c", inp[i]);

    else // If the consonant is uppercase, convert it to lowercase.
      printf(".%c", inp[i] - 'A' + 'a');
  }
  printf("\n");

  return 0;
}