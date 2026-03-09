/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words" using C-style I/O.
 *        This program processes a series of words, abbreviating those
 *        that are excessively long (length > 10) by replacing their
 *        middle characters with a count, while printing shorter words as is.
 *
 * Algorithm: C-style string handling with length check and conditional formatting.
 *            For words longer than 10 characters, it extracts the first character,
 *            calculates the count of intermediate characters (length - 2), and appends
 *            the last character.
 * Time Complexity: O(L) for each word, where L is the length of the word,
 *                  due to string input and basic character access. Total
 *                  time complexity is O(N * L_max) for N test cases.
 * Space Complexity: O(L_max) to store the longest input word.
 */

#include <stdio.h>  // Required for standard input/output functions (scanf, printf)
#include <string.h> // Required for string manipulation functions (strlen)

int main() {
  /**
   * Functional Utility: Manages the execution flow for multiple test cases,
   *                     reading input words and applying the abbreviation logic.
   */
  int n, i, l; // n: number of test cases, i: loop counter, l: length of current word.
  scanf("%d", &n); // Reads the number of test cases from standard input.
  char ch[100];    // Character array (C-style string) to store the current word.
                   // Assumes max word length less than 100.

  /**
   * Block Logic: Processes each test case.
   * Pre-condition: `n` holds the number of test cases.
   * Invariant: `i` iterates from 0 to `n-1`; the loop continues until all
   *            test cases are processed.
   */
  for (i = 0; i < n; i++) {
    scanf("%s", &ch); // Reads the word for the current test case into `ch`.
    l = strlen(ch);   // Calculates the length of the current word.

    /**
     * Block Logic: Checks the length of the current word and applies conditional formatting.
     * Pre-condition: `ch` contains the word; `l` is its length.
     * Invariant: Either the original word or its abbreviated form is printed.
     */
    if (l <= 10) {
      printf("%s\n", ch); // Prints the word as is if its length is 10 or less.
    } else {
      // Prints the abbreviated form: first character, length-2, last character.
      printf("%c", ch[0]);         // Print first character.
      printf("%d", l - 2);         // Print numerical count of middle characters.
      printf("%c\n", ch[l - 1]); // Print last character, followed by a newline.
    }
  }

  return 0; // Indicates successful program termination.
}