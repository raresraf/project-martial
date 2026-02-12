/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem using C-style strings and I/O.
 * @details This program reads an integer `n`, followed by `n` words. It then processes each word.
 * If a word's length is greater than 10, it prints a shortened version consisting of the
 * first character, the number of characters between the first and last, and the last character.
 * Otherwise, it prints the original word. This version stores all words in memory before processing them.
 *
 * @algorithm String abbreviation based on length.
 * @complexity
 * Time complexity: O(N * L), where N is the number of words and L is the maximum length of a word.
 * This is dominated by reading the input and then iterating through the strings.
 * Space complexity: O(N * L) to store all `n` words in a 2D character array.
 */
#include <stdio.h>
#include <string.h>

int main() {
  // n: the number of words. i: loop counter. p: stores the length of a word.
  int n, i, p;
  scanf("%d", &n);
  
  // VLA (Variable Length Array) to store n strings, each up to 100 chars + null terminator.
  char s[n][101]; 
  
  /**
   * @brief Reads all words from input and stores them in a 2D char array.
   */
  for (i = 0; i < n; i++)
    scanf("%s", s[i]);

  // This newline is not typically part of the problem's required output.
  printf("\n");
  
  /**
   * @brief Processes each stored word and prints the result.
   */
  for (i = 0; i < n; i++) {
    p = strlen(s[i]);
    // Check if the length of the word is greater than 10.
    if (p > 10) {
      // Print the abbreviated version.
      printf("%c%d%c\n", s[i][0], p - 2, s[i][p - 1]);
    } else {
      // Print the word as is.
      printf("%s\n", s[i]);
    }
  }
  return 0;
}