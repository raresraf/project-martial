/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program abbreviates words that are longer than 10 characters. The
 * abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character.
 *
 * Algorithm:
 * 1. Read the number of words to process (n).
 * 2. Loop n times:
 *    a. Read a word into a character array.
 *    b. If the length of the word is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word. The repeated calls to strlen can be inefficient.
 * Space Complexity: O(L) to store the longest word.
 */
#include <stdio.h>
#include <string.h>

int main() {
  char word[100];
  int n, i;
  scanf("%d", &n);
  // Invariant: The loop processes n words, applying the abbreviation rule to each.
  for (i = 0; i < n; i++) {
    scanf("%s", word);
    // Block Logic: Checks if the word's length is greater than 10 and prints
    // either the abbreviation or the original word.
    if (strlen(word) > 10) {
      printf("%c%d%c\n", word[0], strlen(word) - 2, word[strlen(word) - 1]);
    } else {
      printf("%s\n", word);
    }
  }

  return 0;
}