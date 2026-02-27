/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces using C-style I/O.
 * @details The problem requires abbreviating words that are longer than 10
 * characters. The abbreviation is formed by the first character, the number of
 * middle characters, and the last character. Shorter words are printed unchanged.
 *
 * @algorithm
 * 1. Read the number of words to process, n.
 * 2. Loop n times:
 *    a. Read a word into a character array.
 *    b. Calculate the length of the word using `strlen`.
 *    c. If the length is greater than 10, print the abbreviation:
 *       - The first character.
 *       - The length minus 2.
 *       - The last character.
 *    d. Otherwise, print the word as is.
 *
 * @complexity
 * Time Complexity: O(n * L), where n is the number of words and L is the
 * maximum length of a word. `strlen` takes O(L) time for each word.
 * Space Complexity: O(L_max), for the character array buffer, where L_max is
 * the maximum possible length of a word (statically set to 100 here).
 */
#include <stdio.h>
#include <string.h>

int main() {
  int n; // Number of words to process.
  int i; // Loop counter.
  int l; // Length of the current word.
  
  scanf("%d", &n);
  
  // A fixed-size character array to store words.
  // Note: This has a fixed limit and is vulnerable to buffer overflows if a
  // word longer than 99 characters is provided.
  char ch[100];

  /**
   * @block
   * @brief Loop through each of the n words.
   */
  for (i = 0; i < n; i++) {
    // Read a word using C-style input.
    scanf("%s", ch);
    
    l = strlen(ch); // Calculate the length of the word.

    /**
     * @brief Check if the word's length exceeds 10 characters.
     */
    if (l <= 10) {
      // If not, print the word as it is.
      printf("%s\n", ch);
    } else {
      // If it's too long, print the abbreviation.
      printf("%c", ch[0]);         // First character
      printf("%d", l - 2);         // Number of middle characters
      printf("%c\n", ch[l - 1]);   // Last character
    }
  }

  return 0;
}
