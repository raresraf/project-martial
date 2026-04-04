/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" can be formed from an input string by deleting zero or more characters.
 *
 * This program implements a sequential scan algorithm to determine if the characters
 * 'h', 'e', 'l', 'l', 'o' appear in order within a given input string, allowing
 * for arbitrary characters to be skipped between the matched characters.
 *
 * Algorithm: Two-pointer scan. One pointer iterates through the input string,
 * the other through the target word "hello".
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(1) for storing constant-size strings.
 */
#include "stdio.h"

int main() {
  /**
   * @brief Main function: Reads an input string and checks for the subsequence "hello".
   * @return 0 if the program executes successfully.
   */
  char hello[5] = {'h', 'e', 'l', 'l', 'o'};

  char str[101];

  scanf("%s", str);

  int i = 0; // Pointer for iterating through the input string 'str'.
  int p = 0; // Pointer for iterating through the target word 'hello'.
  /**
   * @brief Loop through the input string to find the "hello" subsequence.
   * @invariant At the beginning of each iteration, 'p' indicates how many characters
   *            of "hello" have been found in sequence so far.
   */
  while (str[i] != '\0') {
    /**
     * @brief Checks if the current character in 'str' matches the next expected character in "hello".
     * @pre 'i' is within the bounds of 'str', 'p' is within the bounds of 'hello'.
     */
    if (str[i] == hello[p]) {
      p++; // Move to the next character in "hello" to search for.
    }
    /**
     * @brief Checks if all characters of "hello" have been found.
     * @pre 'p' is incremented when a match is found.
     */
    if (p > 4) { // 'hello' has 5 characters, so p > 4 means all 5 are found.
      printf("YES"); // Output "YES" if "hello" is found as a subsequence.
      return 0; // Terminate the program successfully.
    }
    i++; // Move to the next character in the input string.
  }

  printf("NO"); // If the loop finishes and "hello" was not found, output "NO".
  return 0;
}
