/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem.
 * @details This program determines if a user should be classified as male or female
 * based on the number of distinct characters in their username. If the number of
 * distinct characters is even, the user is classified as female ("CHAT WITH HER!").
 * If it's odd, the user is classified as male ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Count the number of unique characters in the string. This is done using an
 *    O(N^2) in-place algorithm:
 *    - Iterate through the string with a pointer 'i'.
 *    - For each character s[i], iterate through the rest of the string (from j=i+1).
 *    - If a duplicate character s[j] == s[i] is found, it's "marked" by
 *      replacing it with the non-printable character with ASCII value 1.
 *    - The outer loop increments a counter for each character it processes, but
 *      decrements it if the character has already been marked as a duplicate by a
 *      previous pass. This effectively counts only the first occurrence of each character.
 * 3. Check if the final count of distinct characters is even or odd and print the
 *    corresponding message.
 *
 * Time Complexity: O(N^2), where N is the length of the string, due to the nested loops.
 * Space Complexity: O(N) to store the input string.
 */

#include <cstdio>

using namespace std;

int main() {
  char s[101]; // Buffer to store the username, max length 100.
  fscanf(stdin, "%s", s);
  int i, j;
  int count = 0; // This will hold the number of distinct characters.

  /**
   * Block Logic: Iterate through the string to find and count unique characters.
   * This implements a brute-force approach to find unique elements.
   */
  for (i = 0; s[i] != '\0'; ++i) {
    /**
     * Pre-condition: s[i] is the character under consideration.
     * Check if this character has already been marked as a duplicate of a
     * character that appeared earlier in the string.
     */
    if (s[i] == 1) {
      --count; // If marked, it's a duplicate, so negate the upcoming increment.
    }

    /**
     * Block Logic: Find all subsequent occurrences of s[i] and mark them.
     * Invariant: After this loop, all characters s[j] (where j > i) that are
     * equal to s[i] will be set to 1.
     */
    for (j = i + 1; s[j] != '\0'; ++j) {
      if (s[j] == s[i]) {
        // Mark the duplicate character with a special value (1).
        s[j] = 1;
      }
    }
    // Increment count for the current character. If it was a duplicate of a
    // character seen before (s[i] == 1), this increment is cancelled out.
    ++count;
  }

  /**
   * Block Logic: Determine the output based on the parity of the distinct character count.
   * An even count corresponds to a female user, odd to a male user.
   */
  if (count % 2 == 0) {
    fprintf(stdout, "CHAT WITH HER!");
  } else {
    fprintf(stdout, "IGNORE HIM!");
  }

  return 0;
}
