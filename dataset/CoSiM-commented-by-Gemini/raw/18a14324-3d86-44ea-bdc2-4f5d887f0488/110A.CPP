/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" competitive programming problem.
 * @details A "lucky number" is a positive integer whose decimal representation contains only the digits 4 and 7.
 * A "nearly lucky number" is a number for which the count of lucky digits (4s and 7s) is itself a lucky number.
 * This program determines if a given number is a nearly lucky number.
 *
 * @section algorithm Algorithm
 * The algorithm is implemented in two distinct phases:
 * 1.  **Count Lucky Digits:**
 *     a. The input number `n` is read.
 *     b. A counter `ct` is initialized to zero.
 *     c. The program iterates through each digit of `n`. This is achieved by repeatedly
 *        taking the number modulo 10 (to get the last digit) and then performing
 *        integer division by 10 (to remove the last digit).
 *     d. In each iteration, if the extracted digit is either 4 or 7, the counter `ct` is incremented.
 * 2.  **Check if the Count is Lucky:**
 *     a. After the loop completes, `ct` holds the total count of lucky digits from the original number.
 *     b. The program then checks if the value of `ct` itself is a lucky number (i.e., if `ct` is equal to 4 or 7).
 *     c. If it is, "YES" is printed. Otherwise, "NO" is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(log10(N)), where N is the input number. The number of iterations in the loop is equal to the number of digits in N.
 * - Space Complexity: O(1), as the program uses a fixed number of variables regardless of the input size.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  long long n; // The input number.
  int ct = 0;  // Counter for the number of lucky digits (4s and 7s).
  cin >> n;
  
  // Phase 1: Count the number of lucky digits in the input number 'n'.
  while (n > 0) {
    // Get the last digit of the number.
    int last_digit = n % 10;
    // Check if the digit is a lucky digit.
    if (last_digit == 4 || last_digit == 7) {
      ct++;
    }
    // Remove the last digit from 'n' to process the next one.
    n /= 10;
  }
  
  // Phase 2: Check if the count of lucky digits ('ct') is itself a lucky number.
  // The only single-digit lucky numbers are 4 and 7. Since the input number can have
  // at most 18 digits, the count can't be a multi-digit lucky number like 44, 47, etc.
  if (ct == 4 || ct == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
