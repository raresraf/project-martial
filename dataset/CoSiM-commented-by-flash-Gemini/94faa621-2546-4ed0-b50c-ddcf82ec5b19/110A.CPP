/**
 * @file 110A.CPP
 * @brief This program determines if a given number is "nearly lucky".
 *        A number is considered nearly lucky if the count of its "lucky digits" (4s and 7s)
 *        is itself a lucky number (i.e., the count is 4 or 7).
 *
 * Algorithm:
 * 1. Read a long long integer 'n' from standard input.
 * 2. Initialize a counter 'c' to 0, which will store the count of lucky digits in 'n'.
 * 3. Iteratively extract each digit of 'n':
 *    a. Get the last digit of 'n' using the modulo operator (n % 10).
 *    b. Remove the last digit from 'n' using integer division (n /= 10).
 *    c. If the extracted digit is 4 or 7, increment 'c'.
 *    d. Continue until 'n' becomes 0.
 * 4. After processing all digits, check if the final count 'c' is equal to 4 or 7.
 * 5. Print "YES" if 'c' is 4 or 7, otherwise print "NO".
 *
 * Time Complexity: O(log10(N)) where N is the input number, as the loop runs
 *                  once for each digit of N.
 * Space Complexity: O(1) as only a few integer variables are used.
 */

#include <bits/stdc++.h>
/**
 * @brief Entry point of the program.
 * Functional Utility: Drives the process of reading an input number,
 * counting its lucky digits, and determining if the number is "nearly lucky".
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares a long long integer 'n' for the input number,
  //              'c' as a counter for lucky digits (initialized to 0),
  //              and 'g' to store each extracted digit.
  long long int n, c = 0, g;
  // Block Logic: Reads the long long integer 'n' from standard input.
  std::cin >> n;
  /**
   * Block Logic: Iteratively extracts digits from 'n' to count occurrences of lucky digits (4 or 7).
   *
   * Invariant: In each iteration, 'g' holds the current last digit of 'n',
   * and 'c' accumulates the count of lucky digits found so far. The loop continues
   * until all digits of 'n' have been processed (i.e., 'n' becomes 0).
   */
  while (n > 0) {
    // Inline: Extracts the last digit of 'n'.
    g = n % 10;
    // Inline: Removes the last digit from 'n'.
    n /= 10;
    // Block Logic: Checks if the extracted digit 'g' is a lucky digit (4 or 7).
    if (g == 7 || g == 4)
      // Inline: Increments the lucky digit counter 'c' if 'g' is a lucky digit.
      c++;
  }
  // Block Logic: Determines and prints whether the original number was "nearly lucky".
  // A number is nearly lucky if its count of lucky digits ('c') is itself a lucky number (4 or 7).
  std::cout << ((c == 7 || c == 4) ? "YES" : "NO");
}
