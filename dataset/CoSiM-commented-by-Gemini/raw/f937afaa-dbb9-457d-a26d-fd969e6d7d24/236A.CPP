/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * Problem Statement:
 * Determine if a user is male or female based on the number of distinct
 * characters in their username. If the count is odd, the user is male ("IGNORE HIM!").
 * If the count is even, the user is female ("CHAT WITH HER!").
 *
 * Algorithm:
 * This program attempts to solve the problem in two stages:
 * 1. Sorting: It first sorts the characters of the input string in descending
 *    order using a selection sort algorithm.
 * 2. Counting: It then iterates through the sorted string to count the number
 *    of distinct characters. The logic relies on the fact that after sorting,
 *    all identical characters are adjacent. It counts a character only when it
 *    is different from the previously seen character.
 *
 * Note on Implementation:
 * The implementation has a critical bug. The variable 'temp' in the second loop
 * is used in a comparison before it is initialized, leading to undefined
 * behavior. The result of the count can be incorrect depending on the garbage
 * value 'temp' happens to hold.
 *
 * Time Complexity: O(L^2), where L is the length of the string, dominated by the
 * selection sort algorithm.
 * Space Complexity: O(L_max), for the character array storage.
 */
#include <stdio.h>
#include <string.h>
int main() {
  int i, j, small, count = 0;
  char temp, temp1, ch[200], ch1[200];
  scanf("%s", ch);

  /**
   * Block Logic: Stage 1 - Sort the character array in descending order.
   * This is an in-place selection sort. After this loop, 'ch' will be sorted.
   */
  for (i = 0; i < strlen(ch); i++) {
    for (j = i + 1; j < strlen(ch); j++) {
      if (ch[j] > ch[i]) {
        temp = ch[i];
        ch[i] = ch[j];
        ch[j] = temp;
      }
    }
  }

  // This variable is initialized but never used.
  temp1 = ch[0];

  /**
   * Block Logic: Stage 2 - Count distinct characters in the sorted array.
   * BUG: 'temp' is uninitialized on the first iteration of this loop.
   * The 'if (ch[i] != temp)' comparison results in undefined behavior.
   */
  for (i = 0; i < strlen(ch); i++) {
    // Because identical characters are now adjacent, a change in character
    // indicates a new distinct character.
    if (ch[i] != temp) {
      count++;
      temp = ch[i];
    }
  }

  // Check the parity of the (potentially incorrect) distinct character count.
  if (count % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}
