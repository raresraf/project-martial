/**
 * @file 71A.CPP
 * @brief C-style solution for Codeforces problem 71A, "Way Too Long Words".
 *
 * @description
 * This program abbreviates words that are longer than 10 characters. The
 * abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character. This solution is
 * written in C, using `scanf`/`printf` and C-style string manipulation.
 *
 * @algorithm
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times:
 *    a. Read a word into a character array.
 *    b. Get the length of the word using `strlen`.
 *    c. If the length is greater than 10, print the abbreviation:
 *       - The first character.
 *       - The length minus 2.
 *       - The last character.
 *    d. Otherwise, print the word as is.
 *
 * @note This implementation includes unnecessary hardcoded special cases for
 * "localization" and "internationalization". The general abbreviation logic
 * in the final `else` block correctly handles these words anyway, making the
 * special checks redundant.
 *
 * @time_complexity O(N * L), where N is the number of words and L is their
 *   maximum length, dominated by the `scanf` and `strlen` calls inside the loop.
 * @space_complexity O(L) for the character array.
 */
#include <cstdio>
#include <cstring>
char x[101];

using namespace std;

int main() {
  int n, i;
  scanf("%d", &n);

  // Block Logic: Process each of the 'n' test cases.
  for (i = 0; i < n; i++) {
    // Read a word into the character array 'x'.
    scanf("%s", &x);

    // Pre-condition: Check if the word is "too long".
    if (strlen(x) > 10) {
      
      // The following 'if-else if' block is redundant. The final 'else'
      // provides the correct general solution for all long words.
      if (!strcmp(x, "localization")) {
        printf("l10n\n");
      } else if (!strcmp(x, "internationalization")) {
        printf("i18n\n");
      } else {
        // General case for abbreviation.
        printf("%c", x[0]);
        printf("%d", strlen(x) - 2);
        printf("%c\n", x[strlen(x) - 1]);
      }
    } else
      // If the word is not too long, print it unchanged.
      printf("%s\n", x);
  }
  return 0;
}
