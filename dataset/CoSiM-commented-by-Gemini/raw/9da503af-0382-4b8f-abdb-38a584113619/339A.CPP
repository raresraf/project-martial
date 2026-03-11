/**
 * @file 339A.CPP
 * @brief Sorts a given mathematical sum of single-digit numbers.
 *
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+2+1") and rearranges it into non-decreasing order (e.g., "1+2+3").
 *
 * @algorithm The program uses a somewhat convoluted approach:
 * 1. It reads the entire sum as a string.
 * 2. It creates an integer array `in`. It iterates through the input string, and
 *    for each character that is a digit, it converts it to an integer and places
 *    it in the `in` array at the same index. The positions corresponding to '+'
 *    are left as default values (0 for a global array).
 * 3. It sorts the initial part of the `in` array, which now contains the digits
 *    and zeros.
 * 4. It iterates through the sorted array, printing the non-zero elements
 *    (the digits) separated by '+'.
 *
 * @note While this logic happens to work for the problem's constraints, it's not
 * a robust or clear way to solve the problem. A more standard approach would be
 * to extract only the digits into a `std::vector`, sort the vector, and then
 * print the elements with '+' separators.
 *
 * @time_complexity O(L log L), where L is the length of the input string, dominated
 * by the sort operation on the integer array.
 * @space_complexity O(L) to store the integer array. The fixed size of 1000 is
 * chosen to be larger than any expected input string length.
 */
#include <bits/stdc++.h>
using namespace std;

// A global array to store the parsed integers from the input string.
int in[1000];

int main() {
  string s;
  cin >> s;
  int len = s.length();
  int j;

  /**
   * @brief Loop to parse the string into an integer array.
   * @invariant The `in` array is populated with digits from the string `s`.
   * Positions of '+' characters are skipped, leaving the default 0 value.
   */
  for (j = 0; j < len; j++) {
    if (s[j] != '+')
      // Convert character digit to integer and store it.
      in[j] = s[j] - 48;
  }

  // Sort the integer array. This brings all the numbers to the end, after the zeros.
  sort(in, in + len);

  int j1;
  /**
   * @brief Loop to print the sorted sum.
   * @invariant The numbers are printed in non-decreasing order.
   */
  for (j1 = 0; j1 < len; j1++) {
    // If the element is not zero (i.e., it was a digit, not a '+').
    if (in[j1] != 0) {
      printf("%d", in[j1]);
      // Print '+' between numbers, but not after the last one.
      if (j1 == len - 1)
        printf("\n");
      else
        printf("+");
    }
  }
  return 0;
}
