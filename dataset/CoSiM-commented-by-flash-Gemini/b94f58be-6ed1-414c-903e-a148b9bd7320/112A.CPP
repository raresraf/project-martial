/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (Problem 112A).
 * @details This program compares two strings lexicographically, ignoring differences in case.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is
 * larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from the input.
 * 2. Convert both strings to a common case (in this implementation, uppercase) to ensure
 *    the comparison is case-insensitive.
 * 3. Iterate through the strings character by character, from left to right.
 * 4. At the first position where the characters differ, determine which string is
 *    lexicographically larger and print the corresponding result (1 or -1), then terminate.
 * 5. If the loop completes without finding any differing characters, the strings
 *    are equal, so print 0.
 *
 * @time_complexity O(L), where L is the length of the strings, as it requires a single pass.
 * @space_complexity O(L) to store the two input strings.
 */
#include <stdio.h>

int main(void) {

  // ASCII character code comments:
  //     A 65 ~ Z 90
  //     a 97 ~ z 122

#define ASC_CHNG 32 // The difference between a lowercase and uppercase letter in ASCII.
  char str[2][100];

  /**
   * @block_logic Read two strings and convert them to uppercase in place.
   */
  for (int i = 0; i < 2; i++) {
    scanf("%s", str[i]);

    for (int j = 0; str[i][j] != '\0'; j++) {
      if (str[i][j] >= 97) { // Check if the character is lowercase.
        str[i][j] = str[i][j] - ASC_CHNG; // Convert to uppercase.
      }
    }
  }

  /**
   * @block_logic Perform character-by-character comparison of the two uppercase strings.
   */
  for (int j = 0; str[0][j] != '\0'; j++) {
    if (str[0][j] > str[1][j]) {
      printf("1");
      return 0;
    }
    if (str[0][j] < str[1][j]) {
      printf("-1");
      return 0;
    }
  }

  // If the loop completes, the strings are identical.
  printf("0");
  return 0;
}
