/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (Codeforces 112A).
 *
 * @details
 * This program performs a case-insensitive lexicographical comparison of two strings
 * using C-style character arrays and manual character manipulation. The goal is to
 * output -1, 1, or 0 based on whether the first string is less than, greater than,
 * or equal to the second.
 *
 * The implementation follows a two-pass approach:
 * 1. Normalization: It reads two strings and converts both of them to uppercase in-place.
 *    This is done by checking the ASCII value of each character and subtracting 32
 *    if it's a lowercase letter.
 * 2. Comparison: It performs a character-by-character comparison of the two now-uppercase
 *    strings. The program terminates and prints the result as soon as the first
 *    differing character is found. If the strings are identical, it prints 0 after
 *    the loop.
 *
 * Algorithm:
 * 1. Read two strings into a 2D char array.
 * 2. Convert both strings to uppercase.
 * 3. Compare the strings character by character.
 * 4. Print the result and exit upon finding the first difference.
 * 5. Print 0 if no differences are found.
 *
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L), to store the input strings.
 */
#include <stdio.h>

int main(void) {

  // ASCII character code ranges for reference
  //     A 65 ~ Z 90
  //     a 97 ~ z 122

// The ASCII difference between a lowercase and uppercase letter is 32.
#define ASC_CHNG 32
  char str[2][100];

  /**
   * @brief Data Ingestion and Normalization Phase.
   * @details This block reads two strings and converts them to uppercase in-place.
   */
  for (int i = 0; i < 2; i++) {
    scanf("%s", str[i]);

    // This inner loop iterates through the string just read, converting it to uppercase.
    for (int j = 0; str[i][j] != '\0'; j++) {
      if (str[i][j] >= 97) { // Check if the character is lowercase.
        str[i][j] = str[i][j] - ASC_CHNG;
      }
    }
  }

  /**
   * @brief Comparison Phase.
   * @details This loop compares the two normalized (now uppercase) strings character
   * by character to determine their lexicographical order.
   */
  for (int j = 0; str[0][j] != '\0'; j++) {
    if (str[0][j] > str[1][j]) {
      printf("1");
      return 0; // Exit as soon as the order is determined.
    }
    if (str[0][j] < str[1][j]) {
      printf("-1");
      return 0; // Exit as soon as the order is determined.
    }
  }

  // If the loop completes, the strings are identical.
  printf("0");
  return 0;
}