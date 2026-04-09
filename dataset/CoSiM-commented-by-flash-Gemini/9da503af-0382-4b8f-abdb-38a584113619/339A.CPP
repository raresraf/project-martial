/**
 * @file 339A.CPP
 * @brief This program processes a mathematical expression string consisting of single-digit
 * numbers separated by '+' signs (e.g., "3+1+2"). It extracts these numbers, sorts them
 * in non-decreasing order, and then reconstructs and prints the sum with the sorted numbers.
 *
 * Algorithm: String parsing, integer conversion, sorting, and string reconstruction.
 * 1. Read the input string.
 * 2. Iterate through the string, extracting only the digit characters and converting them to integers.
 *    Store these integers in an array. Non-digit characters (like '+') implicitly result in
 *    `0`s in the array if the array is sized by the string length, which are later filtered during printing.
 * 3. Sort the array of extracted numbers.
 * 4. Iterate through the sorted array, printing each non-zero number,
 *    and inserting '+' signs between numbers to reconstruct the sum expression.
 *
 * Time Complexity: O(L log L), where L is the length of the input string.
 * This is primarily due to the sorting step. Parsing and printing are O(L).
 * If N is the number of digits, then it's O(L + N log N). Since N <= L, it simplifies to O(L log L).
 *
 * Space Complexity: O(L) to store the input string and the intermediate integer array.
 *
 * Potential Issue: The `in` array is sized by `len` (the total length of the input string
 * including '+' characters). When `sort` is applied to this array up to `len`,
 * the positions corresponding to '+' characters or uninitialized elements become `0`s
 * and are sorted to the beginning. The printing logic explicitly skips `0`s, which
 * mitigates the visual impact but doesn't resolve the underlying array misuse.
 * A more robust solution would involve storing only digits in a dynamically sized container
 * (e.g., `std::vector<int>`) or explicitly handling the number of digits.
 */

#include <bits/stdc++.h> // Includes iostream, string, algorithm for sort, etc.
using namespace std;

// Functional Utility: A global array to store the extracted single-digit numbers.
// Fixed size of 1000, which implies an upper limit on the input string length or number of digits.
int in[1000];

/**
 * @brief Main function to sort and print a sum expression.
 * Reads an arithmetic expression string, extracts numbers, sorts them,
 * and prints the sorted expression.
 * @pre Standard input contains a single string `s` representing a sum of
 *      single-digit numbers (e.g., "1+2+3").
 * @post Prints the sorted sum expression to standard output.
 * @return 0 upon successful execution.
 */
int main() {
  string s;
  // Functional Utility: Reads the input sum expression string.
  cin >> s;
  // Functional Utility: Calculates the length of the input string.
  int len = s.length();
  int j; // Loop counter for parsing.

  /**
   * Block Logic: Parses the input string to extract single-digit numbers.
   * Digits are converted to integers and stored in the `in` array.
   * Characters that are not digits (expected to be '+') are implicitly skipped,
   * leaving `0`s in `in` array at those positions or if the array was not fully filled with digits.
   * Invariant: After each iteration `j`, `in[j]` contains the integer value of `s[j]` if it's a digit,
   *            otherwise its value is undefined or 0 from previous state.
   * Pre-condition: `s` is a valid string of single-digit numbers separated by '+'.
   */
  for (j = 0; j < len; j++) {
    if (s[j] != '+') {
      // Inline: Converts the character digit (ASCII) to its integer equivalent.
      in[j] = s[j] - '0'; // Using '0' instead of 48 for clarity and portability.
    }
    // Else branch is implicitly `in[j]` remains 0 if not assigned,
    // assuming `in` was globally initialized to zeros.
  }

  // Functional Utility: Sorts the array `in` in non-decreasing order.
  // Note: This sort will include any `0`s that might be present in the `in` array
  // due to '+' characters or unassigned elements. These `0`s will be moved to the beginning.
  sort(in, in + len);

  int j1; // Loop counter for printing.
  /**
   * Block Logic: Reconstructs and prints the sorted sum expression.
   * It iterates through the sorted `in` array, printing only non-zero numbers
   * and inserting '+' signs between them. This implicitly handles the `0`s
   * that were sorted to the beginning of the array.
   * Invariant: `in[0...j1-1]` has been processed and printed (if non-zero).
   */
  for (j1 = 0; j1 < len; j1++) {
    // Block Logic: Only process and print if the number is not zero.
    // This effectively skips the `0`s that resulted from '+' characters
    // or uninitialized array parts during the parsing phase.
    if (in[j1] != 0) {
      printf("%d", in[j1]); // Functional Utility: Prints the current digit.
      // Block Logic: Decides whether to print a '+' sign or a newline.
      // Prints '+' if there are more non-zero digits to follow.
      // Prints '\n' if this is the last non-zero digit.
      if (j1 == len - 1)
        printf("\n"); // Inline: Prints a newline character after the last digit.
      else
        printf("+"); // Inline: Prints a '+' sign to separate numbers.
    }
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
