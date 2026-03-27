/**
 * @file 281A.CPP
 * @brief This program reads a stream of words and capitalizes the first letter of each word.
 *        It uses the `toupper` function to convert the first character to uppercase,
 *        leaving other characters unchanged. The processing continues until end-of-file.
 *
 * Algorithm:
 * 1. Read words one by one from standard input until end-of-file.
 * 2. For each word read:
 *    a. Apply the `toupper` function to its first character.
 *    b. Print the modified word to standard output, followed by a newline.
 *
 * Time Complexity: O(L_total) where L_total is the total number of characters across all input words.
 * Space Complexity: O(L_max) where L_max is the length of the longest input word (for string storage).
 */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Continuously reads words from standard input, capitalizes
 * their first letter, and prints them to standard output.
 * @return 0 on successful execution.
 */
int main() {
  string inp;
  /**
   * Block Logic: Reads words from standard input one by one until the end of the input stream (EOF).
   * For each word read, it processes the first character for capitalization and then prints the word.
   *
   * Invariant: Each iteration processes one word. The loop terminates when no more input is available.
   */
  while (cin >> inp) {
    /**
     * Block Logic: Converts the first character of the current word to its uppercase equivalent.
     * Functional Utility: Ensures the first character is capitalized, regardless of its initial case.
     */
    inp[0] = toupper(inp[0]); // Inline: Converts the character to its uppercase equivalent using a standard library function.
    // Block Logic: Prints the modified word to standard output, followed by a newline.
    cout << inp << endl;
  }
}
