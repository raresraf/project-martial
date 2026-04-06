/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - "Nearly Lucky Number".
 * This program determines if a given number is "nearly lucky". A number is considered
 * lucky if all its digits are either 4 or 7. A number is "nearly lucky" if the
 * count of its lucky digits (4s and 7s) is itself a lucky number.
 *
 * @algorithm
 * 1. The `islucky` function checks if a given number consists only of digits 4 and 7.
 *    It iterates through the digits of the number, returning false if any digit is not 4 or 7.
 * 2. The `main` function reads an input number.
 * 3. It then iterates through the digits of the input number to count how many times
 *    the digits 4 or 7 appear.
 * 4. Finally, it calls the `islucky` function with this count. If the count itself
 *    is a lucky number, it prints "YES"; otherwise, it prints "NO".
 *
 * Time Complexity: O(log N) where N is the input number, due to digit extraction.
 * Space Complexity: O(1)
 */
// codeforces110a.cpp
#include <iostream>
using namespace std;
/**
 * @brief Determines if a given number is a "lucky number".
 * A number is considered lucky if all its digits are either 4 or 7.
 * @param n The number to be checked.
 * @return `true` if the number `n` consists only of digits 4 and 7, `false` otherwise.
 *
 * @algorithm
 * Iteratively extracts each digit of the number `n` starting from the least significant.
 * If any digit is found to be neither 4 nor 7, the function immediately returns `false`.
 * If all digits are checked and conform to the lucky digit criteria, the function returns `true`.
 *
 * @pre The input `n` must be a non-negative integer.
 */
bool islucky(long long int n) {
  bool t = n;
  while (n > 0) {
    if (n % 10 != 4 && n % 10 != 7) {
      t = false;
      break;
    }
    n = n / 10;
  }
  return t;
}
int main(int argc, char const *argv[]) {
  /**
   * @brief Main function to solve the "Nearly Lucky Number" problem.
   * Reads an integer, counts its lucky digits (4s and 7s), and then determines
   * if this count itself is a lucky number.
   * @param argc The number of command-line arguments.
   * @param argv An array of command-line argument strings.
   * @return 0 on successful execution.
   */
  long long int n; ///< @var n The input number from which lucky digits will be counted.
  long long int b; ///< @var b A temporary variable used for digit extraction without altering `n`.
  long long int d = 0; ///< @var d Counts the occurrences of lucky digits (4 or 7) in the input number `n`.

  // Read the input number.
  cin >> n;
  b = n; // Initialize 'b' with the input number for digit processing.

  /**
   * @brief Loop to count lucky digits (4s and 7s) in the input number.
   * @invariant `b` progressively reduces as its digits are processed. `d` accumulates
   * the count of lucky digits found so far.
   * @post `d` will contain the total count of lucky digits in the original `n`.
   */
  while (b) {
    /**
     * @brief Conditional check to see if the current least significant digit is lucky.
     * @details Checks if the last digit of `b` is either 4 or 7.
     */
    if (b % 10 == 4 || b % 10 == 7)
      d++; // Increment lucky digit count if condition is met.
    b /= 10; // Remove the least significant digit from `b`.
  }

  /**
   * @brief Determines and prints if the count of lucky digits is a lucky number.
   * @details Calls the `islucky` function with the accumulated count `d`.
   * Prints "YES" if `d` is lucky, otherwise "NO".
   */
  if (islucky(d))
    cout << "YES";
  else
    cout << "NO";
  return 0;
}