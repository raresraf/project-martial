/**
 * @file 71A.CPP
 * @brief Implements an automated word-shortening algorithm (numeronym generation).
 * 
 * Logic: Identifies words with length > 10 and compresses them into a standard 
 * numeronym format (first character + count of internal characters + last character).
 * 
 * Time Complexity: $O(N \times L)$ where N is the number of words and L is the max word length.
 * Space Complexity: $O(L)$ for the character buffer.
 */

#include <stdio.h>
#include <string.h>

int main(void) {
  // Acts as a character buffer for input words.
  char arr[100];
  int n, i, j;

  // Reads the total number of test cases/words to process.
  scanf("%d", &n);

  /**
   * Block Logic: Iteratively processes each input word to apply conditional compression.
   * Invariant: Each word is processed independently without state carry-over.
   */
  for (i = 1; i <= n; i++) {
    scanf("%s", arr);
    // Functional Utility: Check if the word length exceeds the threshold for compression.
    if (strlen(arr) > 10)
      // Generates a numeronym by extracting boundary characters and calculating the middle count.
      printf("%c%d%c\n", arr[0], strlen(arr) - 2, arr[strlen(arr) - 1]);
    else
      // For short words, the original sequence is preserved.
      puts(arr);
  }
  return 0;
}
