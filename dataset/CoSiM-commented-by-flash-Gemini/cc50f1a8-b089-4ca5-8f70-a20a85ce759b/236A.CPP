/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 *
 * This program attempts to determine if a username (string) represents a "boy" or a "girl"
 * based on the parity of the number of distinct characters in the username.
 * If the count of distinct characters is odd, the output should be "IGNORE HIM!".
 * If the count is even, the output should be "CHAT WITH HER!".
 *
 * Algorithm (Intended): Count distinct characters.
 * Time Complexity (Observed): O(L^2) where L is the length of the string, due to nested loops.
 * Space Complexity: O(L) for the input string.
 */
#include <cstdio>

using namespace std;

/**
 * @brief Main function to classify a username as "boy" or "girl".
 *
 * Functional Utility: Reads a username, attempts to count its distinct characters,
 * and prints a message based on the parity of this count.
 */
int main() {
  char s[101];
  fscanf(stdin, "%s", s); // Read the username string from standard input.
  int i, j, count = 0; // Initialize loop counters and distinct character count.

  /**
   * Block Logic (Intended): Iterates through the string to count distinct characters.
   * For each character, it marks subsequent occurrences of that character to avoid recounting.
   * Note: The implementation uses integer '1' to mark characters, which is problematic for char arrays.
   * This approach relies on ASCII value '1' (SOH) and may not behave as expected for all inputs or environments.
   * Pre-condition: 's' contains the input username string.
   * Invariant (Intended): 'count' represents the number of unique characters processed so far.
   */
  for (i = 0; s[i] != '\0'; ++i) {
    // If the current character has been previously marked as '1' (a problematic attempt to skip duplicates).
    // This condition likely causes incorrect behavior as '1' is an ASCII value, not a flag.
    if (s[i] == 1) {
      --count; // Attempts to decrement count for a character already marked as processed.
    }
    /**
     * Block Logic (Intended): Inner loop to mark subsequent occurrences of the current character.
     * This aims to ensure each unique character is counted only once.
     * Note: Setting s[j] = 1 changes the character to ASCII SOH, potentially corrupting the string
     * or causing incorrect comparisons later.
     */
    for (j = i + 1; s[j] != '\0'; ++j) {
      if (s[j] == s[i]) { // If a duplicate of the current character is found.
        s[j] = 1; // Mark the duplicate character (problematic implementation).
      }
    }
    ++count; // Increment count for the current (intended to be unique) character.
  }
  /**
   * Block Logic: Determines if the count of distinct characters is even or odd.
   * Prints "CHAT WITH HER!" if count is even, indicating a "boy".
   * Prints "IGNORE HIM!" if count is odd, indicating a "girl".
   * Post-condition: One of two predefined messages is printed to standard output.
   */
  if (count % 2 == 0) {
    fprintf(stdout, "CHAT WITH HER!");
  } else {
    fprintf(stdout, "IGNORE HIM!");
  }

  return 0;
}
