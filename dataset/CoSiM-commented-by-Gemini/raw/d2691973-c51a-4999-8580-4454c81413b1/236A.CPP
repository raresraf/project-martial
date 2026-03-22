/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces (Problem 236A).
 * @details The program determines a user's gender based on the number of
 * distinct characters in their username. If the number of distinct characters
 * is odd, the user is classified as male ("IGNORE HIM!"). If it's even,
 * the user is female ("CHAT WITH HER!").
 *
 * @section algorithm
 * Algorithm: Frequency Counting
 * 1. An integer array of size 26 is used as a frequency map for the lowercase
 *    English alphabet, initialized to all zeros.
 * 2. The program reads a username string.
 * 3. It iterates through the string, incrementing the count for each character
 *    in the frequency map.
 * 4. It then counts how many entries in the frequency map are non-zero. This
 *    yields the number of distinct characters.
 * 5. The parity of this count is checked to produce the final output.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the username string.
 *   This is dominated by the loop that iterates through the string.
 * - Space Complexity: O(1), as the storage used (the frequency map and a
 *   small buffer) is constant regardless of the input string length.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int num[26], i;

  /**
   * @brief Initializes the frequency map for the alphabet to zero.
   * This array `num` will store counts of each character 'a' through 'z'.
   */
  for (i = 0; i < 26; i++) {
    num[i] = 0;
  }

  char arr[101];
  cin.getline(arr, 101);
  int co = 0, x, len = strlen(arr);

  /**
   * @brief Populates the frequency map based on the input string.
   * @invariant `num[k]` holds the number of times the k-th letter of the
   * alphabet has appeared in the processed part of the string.
   */
  for (i = 0; i < len; i++) {
    x = arr[i] - 'a'; // Maps character to an index 0-25.
    num[x]++;
  }

  /**
   * @brief Counts the number of distinct characters found in the string.
   */
  for (i = 0; i < 26; i++) {
    if (num[i] > 0) {
      co++;
    }
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * According to the problem statement, an even count implies the user is female,
   * and an odd count implies the user is male.
   */
  if (co % 2 == 0) {
    printf("CHAT WITH HER!\n");
  } else {
    printf("IGNORE HIM!\n");
  }
}
