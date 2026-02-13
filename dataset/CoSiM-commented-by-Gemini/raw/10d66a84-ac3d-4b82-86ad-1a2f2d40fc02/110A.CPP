/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem (Codeforces 110A).
 * @author competitive-programmer
 *
 * @details This program determines if a number is "nearly lucky". A number is
 * nearly lucky if the count of its "lucky digits" ('4' and '7') is itself
 * a lucky number (a positive integer composed only of digits '4' and '7').
 *
 * Algorithm:
 * 1. Count Lucky Digits:
 *    a. Read the input number as a sequence of characters until a non-digit
 *       is encountered.
 *    b. For each character, check if it is '4' or '7'.
 *    c. Increment a counter, `count`, for each lucky digit found.
 *
 * 2. Check if the Count is Lucky:
 *    a. If `count` is 0, the number is not nearly lucky.
 *    b. Otherwise, iterate through the digits of `count`.
 *    c. For each digit of `count`, check if it is '4' or '7'.
 *    d. If any digit is not '4' or '7', the `count` is not a lucky number,
 *       so the original number is not nearly lucky.
 *    e. If all digits of `count` are '4' or '7', the original number is
 *       nearly lucky.
 *
 * Time Complexity: O(log10(N)) to count the lucky digits in the input number N,
 * plus O(log10(C)) where C is the count of lucky digits. Effectively, it's
 * linear in the number of digits of the input.
 * Space Complexity: O(1).
 */
#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  char c;
  int count = 0; // This will store the number of lucky digits ('4' or '7').

  /**
   * Block Logic: Read the input number character by character and count the
   * occurrences of lucky digits.
   */
  while (1) {
    fscanf(stdin, "%c", &c);
    // Stop reading when a non-digit character (like a newline) is found.
    if (!isdigit(c)) {
      break;
    }
    // Check if the current digit is a lucky digit.
    if (c - '0' == 4 || c - '0' == 7) {
      ++count;
    }
  }

  // A number with zero lucky digits is not nearly lucky.
  if (count == 0) {
    fprintf(stdout, "NO");
    return 0;
  }

  int m = 10, n = 1;
  int dig;

  /**
   * Block Logic: Check if the 'count' itself is a lucky number.
   * This is done by iterating through each digit of 'count'.
   */
  while (count / n > 0) {
    // Isolate the current digit using modulo and division.
    dig = (count % m) / n;
    
    // Pre-condition: If any digit is not 4 or 7, it's not a lucky number.
    if (dig != 4 && dig != 7) {
      fprintf(stdout, "NO");
      return 0;
    }
    // Update m and n to check the next digit in the next iteration.
    m = m * 10;
    n = n * 10;
  }
  
  // If the loop completes, it means all digits of 'count' were 4 or 7.
  fprintf(stdout, "YES");
}
