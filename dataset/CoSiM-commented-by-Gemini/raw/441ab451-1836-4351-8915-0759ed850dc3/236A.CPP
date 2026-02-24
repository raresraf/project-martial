/**
 * @file raw/441ab451-1836-4351-8915-0759ed850dc3/236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl. This program
 * determines if the number of distinct characters in a username is even or odd.
 *
 * @algorithm
 * The core logic relies on counting the number of unique characters in the input
 * string. A boolean array `check` of size 26 serves as a hash set to keep
 * track of the characters encountered.
 *
 * 1. Initialize `check` array to all zeros.
 * 2. Read the input username string.
 * 3. Iterate through each character of the string.
 * 4. For each character, calculate its 0-indexed position relative to 'a'.
 * 5. If the character has not been seen before (i.e., `check[position]` is 0),
 *    mark it as seen by setting `check[position]` to 1 and increment the
 *    `distinct_chars` counter.
 * 6. After iterating through the entire string, check the parity of `distinct_chars`.
 * 7. If `distinct_chars` is even, output "CHAT WITH HER!".
 * 8. If `distinct_chars` is odd, output "IGNORE HIM!".
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the username string. This is
 *   because we perform a single pass over the string.
 * - Space Complexity: O(1), as the space used by the `check` array is constant
 *   (26 elements), and the space for the input string is not counted towards
 *   auxiliary space.
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