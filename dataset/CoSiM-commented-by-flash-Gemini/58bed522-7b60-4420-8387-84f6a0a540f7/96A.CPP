/**
 * @58bed522-7b60-4420-8387-84f6a0a540f7/96A.CPP
 * @brief This program checks for the presence of specific characters ('H', 'Q', or '9')
 * in its standard input stream. If any of these characters are found, it indicates
 * a positive result and terminates early. Otherwise, it indicates a negative result
 * after processing all input. This problem is commonly known as "HQ9+" in competitive programming.
 *
 * Algorithm: Linear scan of the input stream with early exit on condition match.
 * Time Complexity: O(L) where L is the number of characters in the input stream (worst case).
 * Space Complexity: O(1) for storing variables.
 */

/* 2012
 * Maciej Szeptuch
 * II UWr
 */
#include <cstdio> // Includes the C standard input/output library for getchar and puts.

int c; // Declares an integer variable 'c' to store the character read from input.

int main(void) { // Main function, returns an integer to indicate success/failure.
  /**
   * Block Logic: Reads characters one by one from standard input until the End-Of-File (EOF) marker is reached.
   * Invariant: 'c' holds the ASCII value of the current character, or EOF.
   */
  while ((c = getchar()) != -1) { // 'getchar()' reads a single character; returns EOF on end of input.
                                   // EOF is typically -1.
    /**
     * Block Logic: Checks if the current character is one of the target characters ('H', 'Q', or '9').
     * Pre-condition: 'c' holds a character value from the input stream.
     * Invariant: If a target character is found, "YES" is printed, and the program exits immediately.
     */
    if (c == 'H' || c == 'Q' || c == '9') { // Inline: Checks if 'c' matches any of the required characters.
      puts("YES"); // Prints "YES" followed by a newline.
      return 0;    // Terminates the program successfully.
    }
  }

  // Block Logic: If the loop finishes, it means none of the target characters were found.
  puts("NO");    // Prints "NO" followed by a newline.
  return 0;      // Terminates the program successfully.
}
