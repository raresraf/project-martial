/**
 * @file 58A.CPP
 * @brief Solution for a subsequence checking problem (e.g., Codeforces 58A "Chat room").
 * @details This program determines if the word "hello" can be formed by deleting zero or
 * more characters from a given input string. This is equivalent to checking if "hello"
 * is a subsequence of the input string.
 *
 * @algorithm Two-Pointer Subsequence Check
 * @complexity
 * - Time: O(L), where L is the length of the input string, as it performs a single
 *   pass over the input.
 * - Space: O(L) to store the input string `str`.
 */
#include "stdio.h"

int main() {
  // The target subsequence to find.
  char hello[5] = {'h', 'e', 'l', 'l', 'o'};

  // Buffer to hold the input string. Assumes max length of 100 characters.
  char str[101];

  // Read the input string using C-style I/O.
  scanf("%s", str);

  // 'i' is a pointer/index for the input string `str`.
  int i = 0;
  // 'p' is a pointer/index for the target `hello` subsequence.
  int p = 0;

  /**
   * @block Subsequence Search Loop
   * @logic Iterates through the input string `str` to find the characters of `hello` in order.
   */
  while (str[i] != '\0') {
    // If the current character in the input matches the current character in the target...
    if (str[i] == hello[p]) {
      // ...then we have found the next character of the subsequence, so we advance
      // the pointer for the target.
      p++;
    }

    // If 'p' exceeds 4, it means we have found all 5 characters of "hello" (indices 0 through 4).
    if (p > 4) {
      printf("YES");
      return 0; // Found it, exit successfully.
    }

    // Always move to the next character in the input string.
    i++;
  }

  // If the loop finishes before finding all characters, "hello" is not a subsequence.
  printf("NO");
  return 0;
}
