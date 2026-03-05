/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem.
 *
 * A "lucky number" is a positive integer whose decimal representation contains
 * only the digits 4 and 7. A number is "nearly lucky" if the count of lucky
 * digits (4s and 7s) in it is a lucky number. This program determines if a
 * given number is nearly lucky.
 *
 * @algorithm
 * The process is done in two main stages:
 *
 * Stage 1: Count the lucky digits in the input number.
 * 1. Read the input number `n`.
 * 2. Initialize a counter `c` to 0.
 * 3. Iterate through each digit of `n`. This is done by repeatedly taking the
 *    number modulo 10 to get the last digit, and then integer-dividing by 10
 *    to remove the last digit.
 * 4. If a digit is either 4 or 7, increment the counter `c`.
 *
 * Stage 2: Determine if the count `c` is itself a lucky number.
 * 1. After Stage 1, `c` holds the total count of lucky digits from `n`.
 * 2. The program then checks if this count `c` is a lucky number. A number is
 *    lucky if all of its digits are either 4 or 7.
 * 3. It iterates through the digits of `c`, counting the total number of digits (`k`)
 *    and the number of digits that are 4 or 7 (`l`).
 * 4. If `l` (count of 4s and 7s in `c`) is not equal to `k` (total digits in `c`),
 *    it means `c` contains other digits and is therefore not a lucky number.
 * 5. An additional check handles the case where the count `c` is 0. A count of 0
 *    is not considered a lucky number.
 * 6. If `c` is determined to be a lucky number (and is not 0), print "YES".
 *    Otherwise, print "NO".
 *
 * @implementation_note
 * The logic to check if the count `c` is a lucky number is overly complex for the
 * constraints of the problem. A much simpler approach would be to directly check
 * if `c` is equal to 4 or 7, as these are the only lucky numbers that can be
 * realistically formed by counting digits in a `long long`.
 *
 * @complexity
 * Time Complexity: O(log10(n)), as it iterates through the digits of the input number `n`
 * and then through the digits of the count `c`.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  long long int n; // The input number.
  long long int i; // A copy of n to be modified.
  long long int c = 0; // Counter for lucky digits (4s and 7s).
  long long int r = 0; // To store the remainder (current digit).
  int m = 1; // A flag to handle the case where the count `c` is 0.

  cin >> n;
  i = n;

  // --- Stage 1: Count lucky digits in the input number n ---
  while (i != 0) {
    r = i % 10; // Get the last digit.
    if (r == 4 || r == 7) {
      c++; // Increment the counter if the digit is lucky.
    }
    i = i / 10; // Remove the last digit.
  }

  // --- Stage 2: Check if the count `c` is a lucky number ---
  
  // Handle the case where no lucky digits were found. 0 is not a lucky number.
  if (c == 0) {
    m = 0;
  }

  long long int k = 0; // Counts total digits in `c`.
  long long int l = 0; // Counts lucky digits (4 or 7) in `c`.
  
  // This loop iterates through the digits of the count `c`.
  while (c != 0) {
    r = c % 10; // Get the last digit of `c`.
    k++;        // Increment total digit count for `c`.
    if (r == 4 || r == 7) {
      l++;    // Increment lucky digit count for `c`.
    }
    c /= 10; // Remove the last digit of `c`.
  }

  // A number is lucky if ALL its digits are lucky digits.
  // So, the total number of digits (`k`) must equal the number of lucky digits found (`l`).
  // Also, the count must not have been zero (`m != 0`).
  if (l == k && m != 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
