/**
 * @file 71A.CPP
 * @brief This program implements the "Way Too Long Words" problem.
 * It abbreviates words longer than 10 characters by replacing the middle characters
 * with the count of characters, keeping the first and last characters.
 * Algorithm: String length check and conditional printing/abbreviation.
 * Time Complexity: O(N * L), where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L) for storing the buffer for each word.
 */

#include <cstdio>
#include <cstring>
using namespace std;

int main() {
  int n; // Stores the number of words to process.
  char buffer[101]; // Buffer to store each word, max length 100 + null terminator.
  int len; // Stores the length of the current word.
  // Reads the number of words that will follow.
  scanf("%d", &n);
  /**
   * Block Logic: Processes 'n' words, one in each iteration.
   * Invariant: Each iteration handles one word according to the abbreviation rules.
   */
  while (n--) {
    // Reads a word into the buffer.
    scanf("%s", buffer);
    // Calculates the length of the read word.
    len = strlen(buffer);
    /**
     * Conditional Logic: Checks if the word needs abbreviation or should be printed as is.
     * If the word's length is greater than 10, it's abbreviated.
     * Otherwise, the original word is printed.
     */
    if (len > 10) {
      // Prints the first character of the word.
      printf("%c", buffer[0]);
      // Prints the number of characters between the first and last (length - 2).
      printf("%d", len - 2);
      // Prints the last character of the word.
      printf("%c", buffer[len - 1]);
    } else {
      // Prints the word as it is, without abbreviation.
      printf("%s", buffer);
    }
    // Prints a newline character after each word (or its abbreviation).
    printf("\n");
  }
  // Standard program exit.
  return 0;
}
