/**
 * @file 339A.CPP
 * @brief Solution to Codeforces Problem 339A: "Helpful Maths".
 *        The problem involves sorting a mathematical sum given as a string.
 *        The sum consists of numbers 1, 2, and 3 separated by '+' signs.
 *        The task is to sort the numbers in non-decreasing order and print
 *        the sum again.
 *
 * Algorithm: String parsing, integer extraction, sorting, and formatted output.
 *            The program reads the input string, extracts the numbers, stores them
 *            in an array, sorts the array, and then prints the numbers back
 *            separated by '+' signs.
 * Time Complexity: O(L + N log N) where L is the length of the input string,
 *                  and N is the number of integers in the string.
 *                  Extraction is O(L), sorting is O(N log N), and printing is O(N).
 * Space Complexity: O(L) to store the input string and the extracted numbers.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * Functional Utility: Declaration of variables for string processing and integer storage.
   * `str`: Character array to store the input sum string (max 104 chars + null terminator).
   * `a`: Integer array to store the extracted numbers (max 105 numbers).
   * `i`, `j`: Loop counters; `j` also tracks the number of extracted integers.
   */
  char str[105];
  int a[105];
  int i, j;
  /**
   * Block Logic: Reads input sum strings until End-Of-File (EOF).
   * This loop allows processing multiple test cases if provided sequentially.
   * Pre-condition: Input stream contains sum strings in the format "N+N+N...".
   * Invariant: `str` contains the current sum string to be processed.
   */
  while (scanf("%s", str) != EOF) {
    /**
     * Block Logic: Parses the input string `str` to extract numerical digits.
     * `i` iterates through `str`, `j` tracks the insertion index in array `a`.
     * Pre-condition: `str` contains a null-terminated string of numbers and '+' signs.
     * Invariant: `a` contains digits extracted from `str` up to index `j-1`.
     */
    for (i = 0, j = 0; i < strlen(str); i++) {
      /**
       * Conditional Logic: Checks if the current character is a digit.
       * If it's a digit, convert it to an integer and store it in array `a`.
       */
      if (str[i] >= '0' && str[i] <= '9') {
        a[j++] = str[i] - '0'; // Convert char digit to int and store.
      }
    }
    // Functional Utility: Sorts the extracted numbers in non-decreasing order.
    // Algorithm: Standard library sort (typically IntroSort, a hybrid algorithm).
    sort(a, a + j);
    /**
     * Block Logic: Prints the sorted numbers back as a sum, separated by '+'.
     * Pre-condition: Array `a` contains `j` sorted integers.
     * Invariant: Numbers from `a[0]` to `a[i-1]` have been printed, with '+' separators.
     */
    for (i = 0; i < j; i++) {
      cout << a[i];
      /**
       * Conditional Logic: Prints a '+' separator if it's not the last number.
       */
      if (i < j - 1)
        cout << "+";
    }
    // Functional Utility: Prints a newline character after each processed sum.
    cout << endl;
  }
  return 0;
}
