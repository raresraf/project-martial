/**
 * @file 71A.CPP
 * @brief Abbreviates long words to a shorter format using C-style strings.
 *
 * This program implements a word abbreviation logic commonly seen in competitive
 * programming. It processes a series of words, and for each word, it checks
 * its length. If the word is longer than 10 characters, it is abbreviated by
 * taking its first character, followed by the count of characters between the
 * first and last, and finally the last character. Words with 10 or fewer
 * characters are left unchanged. This version uses C-style I/O and character arrays.
 *
 * @warning The use of `scanf("%s", arr)` with a fixed-size buffer `char arr[100]`
 * is dangerous and can lead to a buffer overflow if an input word is longer
 * than 99 characters (plus the null terminator).
 *
 * @algorithm String length check and conditional formatting using `strlen` and `printf`.
 * @time_complexity O(T * L), where T is the number of test cases and L is the
 * maximum length of a word. `strlen` takes O(L) time.
 * @space_complexity O(C), where C is the constant size of the character array (100).
 */
#include <stdio.h>
#include <string.h>

int main(void) {
  char arr[100]; // Fixed-size buffer to store the input word.
  int n;         // The number of test cases.
  int i, j;      // Loop counters (j is declared but unused).

  // Read the number of words to process.
  scanf("%d", &n);

  /**
   * @brief Main loop to process each word.
   * @invariant The loop runs `n` times.
   */
  for (i = 1; i <= n; i++) {
    // Read the word into the character array.
    scanf("%s", arr);
    
    /**
     * @brief Conditional block to check if the word needs abbreviation.
     * @pre `arr` contains a null-terminated string.
     */
    if (strlen(arr) > 10) {
      // If the word's length is > 10, print the abbreviated form.
      printf("%c%d%c\n", arr[0], (int)strlen(arr) - 2, arr[strlen(arr) - 1]);
    } else {
      // Otherwise, print the word as is. `puts` adds a newline automatically.
      puts(arr);
    }
  }
  return 0;
}
