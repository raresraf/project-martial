/**
 * @file 96A.CPP
 * @brief Solves the "HQ9+" problem from Codeforces (Problem 133A), not 96A as the filename suggests.
 * @details The problem is to determine if a given program, written in the esoteric language HQ9+,
 * will produce any output. In HQ9+, only the commands 'H', 'Q', and '9' produce output.
 *
 * @algorithm
 * The solution iterates through the input string character by character. It checks if
 * the current character is one of 'H', 'Q', or '9'. If any of these characters are
 * found, the program immediately prints "YES" and terminates, as output is guaranteed.
 * If the entire input is processed without finding any of these characters, the program
 * prints "NO".
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string, as it may need to
 * scan the entire string in the worst case (when no output-producing commands are present).
 * Space Complexity: O(1), as it only uses a single variable to store the current character.
 */
/* 2012
 * Maciej Szeptuch
 * II UWr
 */
#include <cstdio>

/**
 * @variable c
 * @brief Stores the character currently being read from standard input.
 * It's an `int` to accommodate the End-Of-File marker (`-1`).
 */
int c;

int main(void) {
  /**
   * @block
   * @brief Reads the input character by character until the end of the stream is reached.
   * @invariant No 'H', 'Q', or '9' has been found in the stream processed so far.
   */
  while ((c = getchar()) != -1)
    // Pre-condition: Check if the character is an output-producing command.
    if (c == 'H' || c == 'Q' || c == '9') {
      puts("YES"); // Output is guaranteed.
      return 0;    // Terminate immediately.
    }

  // If the loop completes, no output-producing command was found.
  puts("NO");
  return 0;
}
