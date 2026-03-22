/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem (Codeforces 281A) using C-style functions.
 * @details This program reads a single word and capitalizes its first letter,
 * then prints the modified word.
 *
 * @section algorithm
 * Algorithm: In-place Modification
 * 1. A character array (C-style string) is used as a buffer to read the input word.
 * 2. The standard C library function `scanf` is used to read the word.
 * 3. The `toupper` function from `<ctype.h>` is called on the first character
 *    of the array (`ch[0]`) to convert it to uppercase. The result replaces
 *    the original character.
 * 4. The `printf` function is used to print the entire modified string.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the string. This is
 *   dominated by the I/O operations (`scanf`, `printf`). The capitalization
 *   itself is an O(1) operation.
 * - Space Complexity: O(B), where B is the buffer size (1000). A fixed-size
 *   buffer is used to store the string.
 */
#include <ctype.h>
#include <stdio.h>
#include <string.h>

int main() {
  char ch[1000];
  // Reads a single word from standard input into the character array.
  scanf("%s", ch);

  /**
   * @brief Capitalizes the first letter of the string in-place.
   * This uses the standard C library function `toupper` for safe case conversion.
   */
  ch[0] = toupper(ch[0]);

  // Prints the modified string, which now has a capitalized first letter.
  printf("%s", ch);
  
  return 0;
}