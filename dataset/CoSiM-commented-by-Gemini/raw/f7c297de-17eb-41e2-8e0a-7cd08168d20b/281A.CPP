/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (Codeforces 281A).
 *
 * Problem Statement:
 * Given a word, the task is to capitalize its first letter. That is, if the first
 * letter is lowercase, it should be converted to uppercase. All other letters
 * of the word should remain unchanged.
 *
 * Algorithm:
 * The program reads a string from the standard input. It then checks if the first
 * character of the string is a lowercase letter by inspecting its ASCII value.
 * In ASCII, lowercase letters 'a' through 'z' range from 97 to 122. If the first
 * character falls into this range, it is converted to its uppercase equivalent by
 * subtracting 32 (the ASCII difference between corresponding lowercase and
 * uppercase letters). Finally, the modified string is printed to the standard output.
 *
 * Time Complexity: O(L), where L is the length of the input string, dominated by
 * the I/O operations (scanf/printf). The capitalization logic is O(1).
 * Space Complexity: O(L_{max}), where L_{max} is the maximum allowed length of the
 * string (1000 characters in this case), for storing the input string.
 */
#include <stdio.h>

int main(void) {

  // Comments in original code detailing ASCII value ranges.
  //     A 65 ~ Z 90
  //     a 97 ~ z 122
  //

  // The difference between a lowercase and uppercase letter in the ASCII table.
#define ASC_CHNG 32

  // A character array to store the input word.
  char input[1000];

  // Read the word from standard input.
  scanf("%s", input);

  /**
   * Block Logic: Capitalize the first letter if it is lowercase.
   * Pre-condition: 'input' contains the word.
   */
  // Check if the first character's ASCII value is in the lowercase range.
  if (input[0] >= 97) {
    // Convert to uppercase by subtracting the ASCII difference.
    input[0] = input[0] - ASC_CHNG;
  }

  // Print the resulting word.
  printf("%s", input);

  return 0;
}
