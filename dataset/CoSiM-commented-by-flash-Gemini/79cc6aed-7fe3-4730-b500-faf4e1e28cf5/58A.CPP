/**
 * @file 58A.CPP
 * @brief Solution to Codeforces problem 58A - Chat room.
 *        This program determines if the word "hello" can be formed as a subsequence
 *        of a given input string. This means the characters 'h', 'e', 'l', 'l', 'o'
 *        must appear in the input string in that order, but not necessarily consecutively.
 * Algorithm:
 *   1. Reads an input string `s`.
 *   2. Uses five nested loops to search for the characters 'h', 'e', 'l', 'l', 'o' sequentially.
 *      - The outermost loop searches for 'h'.
 *      - The next loop searches for 'e' starting from the position after 'h'.
 *      - This pattern continues for 'l', 'l', and 'o'.
 *   3. If all characters are found in the correct order, "YES" is printed, and the program terminates.
 *   4. If the nested loops complete without finding the complete subsequence, "NO" is printed.
 * Time Complexity: O(N^5) in the worst case, where N is the length of the input string, due to the five nested loops.
 * Space Complexity: O(N) for storing the input string.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

char s[105];
int main(int argc, char **argv) {
  /**
   * @brief Main function implementing the solution for the "Chat room" problem.
   *
   * Reads an input string and searches for the "hello" subsequence using
   * a series of nested loops. It prints "YES" and exits if found, otherwise
   * prints "NO".
   * @param argc Argument count (unused).
   * @param argv Argument vector (unused).
   * @return int Returns 0 upon successful execution if "hello" is not found,
   *             or 0 if "hello" is found (due to early exit).
   */
  scanf("%s", s); // Reads the input string into character array 's'.
  int len = strlen(s); // Gets the length of the input string.

  // Block Logic: Nested loops to find 'h', 'e', 'l', 'l', 'o' in sequence (subsequence search).
  // The structure iterates through all possible starting positions for each character,
  // ensuring that subsequent characters are found *after* the previous one.
  // This approach has a time complexity of O(len^5) in the worst case.
  
  // Outer loop searches for 'h'.
  for (int i = 0; i < len; ++i) {
    if (s[i] == 'h') // Found 'h'
      // Second loop searches for 'e' after 'h'.
      for (int j = i + 1; j < len; ++j) {
        if (s[j] == 'e') // Found 'e'
          // Third loop searches for first 'l' after 'e'.
          for (int k = j + 1; k < len; ++k) {
            if (s[k] == 'l') // Found first 'l'
              // Fourth loop searches for second 'l' after first 'l'.
              for (int a = k + 1; a < len; ++a) {
                if (s[a] == 'l') // Found second 'l'
                  // Innermost loop searches for 'o' after second 'l'.
                  for (int b = a + 1; b < len; ++b) {
                    if (s[b] == 'o') { // Found 'o'
                      puts("YES");   // All characters of "hello" found as subsequence.
                      return 0;      // Exit successfully.
                    }
                  }
              }
          }
      }
  }
  puts("NO"); // If execution reaches here, "hello" subsequence was not found.
  return 0;
}
