/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem using a counting sort approach.
 *
 * @details
 * The program reorders a string representing a sum (e.g., "3+2+1") into a sum
 * with non-decreasing terms (e.g., "1+2+3"). This implementation is highly
 * efficient as it uses a frequency array (counting sort) instead of a general-
 * purpose sorting algorithm.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a frequency array `n` to store the counts of each digit ('1', '2', '3').
 * 3. Iterate through the input string and count the occurrences of each digit,
 *    updating the frequency array. For a digit `d`, `n[d]` is incremented.
 * 4. Reconstruct the sorted string:
 *    a. Iterate through the digits in increasing order (1, 2, 3).
 *    b. For each digit, print it as many times as it was counted, separated by '+'.
 *    c. A flag is used to ensure '+' is only printed before the second and
 *       subsequent terms.
 *
 * Time Complexity: O(L), where L is the length of the input string. This is because
 *                  the counting and printing phases are both linear in time.
 * Space Complexity: O(1), as the frequency array has a small, constant size.
 */
#include <bits/stdc++.h>
using namespace std;
// Global array to store the frequency of digits 1, 2, and 3.
int n[5];
int main() {
  // freopen("input.txt", "r", stdin);
  string s;
  cin >> s;
  /**
   * @brief Counts the occurrences of each digit ('1', '2', '3') in the string.
   * The expression `s[i] - '0'` converts a digit character to its integer value.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      n[s[i] - '0']++;
  }
  bool f = false; // Flag to control the printing of the '+' separator.
  /**
   * @brief Reconstructs and prints the sorted sum.
   * It iterates from 1 to 3 and prints each digit `n[i]` times.
   */
  for (int i = 1; i < 3 + 1; i++) {
    // Loop as many times as the digit `i` appeared in the input.
    while (n[i]--) {
      // Print a '+' before every term except the very first one.
      if (f)
        printf("+");
      f = true; // Set the flag after the first term is ready to be printed.
      printf("%d", i);
    }
  }
  printf("\n");
  return 0;
}
