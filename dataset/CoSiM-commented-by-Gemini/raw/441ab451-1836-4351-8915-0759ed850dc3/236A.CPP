/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision logic is based on the number of distinct characters in a
 * given username. If the count of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If it's even, the user is female
 * ("CHAT WITH HER!").
 *
 * This solution uses a C-style approach with a frequency array to count distinct
 * characters, which is very efficient for a small, known character set like
 * lowercase English letters.
 *
 * The algorithm is as follows:
 * 1. An integer array `check[26]` is used as a boolean flag array (or hash set)
 *    for the 26 lowercase letters.
 * 2. Read the username into a C-style character array.
 * 3. Iterate through the input string.
 * 4. For each character, calculate its 0-based index (`x[i] - 'a'`).
 * 5. Use this index to check the `check` array. If the character has already been
 *    seen (`check[index] == 1`), do nothing.
 * 6. If it's a new character, mark it as seen in the `check` array and increment
 *    a counter `n` for distinct characters.
 * 7. After the loop, check the parity of `n` and print the correct message.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string, due to the single pass.
 * - Space Complexity: O(1), as the `check` array has a constant size (26),
 *   plus O(L) for the input string itself.
 */
#include <cstdio>

using namespace std;

// Frequency array to track seen characters. Indexed by 'a' through 'z'.
int check[26];

int main() {
  int n = 0, i, temp; // `n` will count distinct characters.
  char x[101];
  scanf("%s", &x);
  /**
   * @brief Iterates through the string to count distinct characters using a frequency array.
   */
  for (i = 0; x[i] != 0; i++) {
    temp = x[i] - 'a'; // Calculate 0-based index for the character.
    
    // If we have already counted this character, skip.
    if (check[temp] == 1) {
      continue;
    } else {
      // Otherwise, mark it as seen and increment the distinct count.
      check[temp] = 1;
      n++;
    }
  }
  /**
   * @brief Determines the output based on the parity of the distinct character count.
   */
  if (n % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}