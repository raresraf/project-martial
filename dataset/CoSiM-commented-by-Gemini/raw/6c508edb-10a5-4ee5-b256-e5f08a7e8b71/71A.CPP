
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A, "Way Too Long Words".
 * @details This program abbreviates words that are longer than 10 characters using
 * C-style string manipulation and dynamic memory allocation.
 *
 * Algorithm:
 * For each of the `n` input words, the program dynamically allocates a character
 * buffer. It reads the word into the buffer, checks its length using `strlen`.
 * If the length is greater than 10, it prints an abbreviation consisting of the
 * first character, the number of middle characters (length - 2), and the last
 * character. Otherwise, it prints the original word. After processing, the
 * allocated memory for the word is freed.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is their
 * maximum length. `strlen` is called multiple times per word.
 * Space Complexity: O(L) for the dynamically allocated buffer for the longest word.
 */
int main(void) {
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    /**
     * @brief Dynamically allocate memory for a C-style string.
     * Implementation Note: `sizeof(word)` is used instead of `sizeof(char)`.
     * Since `word` is a `char*`, this allocates `101 * sizeof(pointer)` bytes,
     * which is more than intended but sufficient.
     */
    char *word = (char *)malloc(101 * sizeof(word));

    cin >> word;

    /**
     * @brief Abbreviation logic block.
     * Pre-condition: `word` contains the null-terminated input string.
     * Logic: If word length > 10, print abbreviation, else print the word.
     */
    if (strlen(word) > 10) {
      // Inefficiently calls strlen three times, but the logic is correct.
      cout << word[0] << strlen(word) - 2 << word[strlen(word) - 1] << endl;
    } else {
      cout << word << endl;
    }

    // Deallocate the memory for the current word.
    free(word);
  }

  return 0;
}
