/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 *
 * This program determines if a given number is a "nearly lucky number".
 * A number is nearly lucky if the count of its "lucky digits" (4 and 7)
 * is itself a lucky number (i.e., either 4 or 7).
 *
 * Algorithm: Digit extraction and counting. The program repeatedly extracts
 * the last digit of the input number, checks if it's 4 or 7, and increments
 * a counter. This process continues until the number becomes 0. Finally,
 * it checks if the count is 4 or 7.
 * Time Complexity: O(log N) where N is the input number, as it iterates
 * once for each digit.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
/**
 * @brief Main function to determine if an input number is a "nearly lucky number".
 *
 * Reads an integer, counts the occurrences of lucky digits (4 and 7),
 * and then checks if this count is itself a lucky number.
 *
 * @return void (standard competitive programming practice; implicitly returns 0 on successful exit).
 */
int main() {
  // Declare an integer 'n' to store the input number.
  int n;
  // Read the input number into 'n'.
  cin >> n;
  // Declare an integer 'count' to store the number of lucky digits, initialized to 0.
  int count = 0;
  /**
   * @brief Block Logic: Extracts digits and counts lucky ones.
   *
   * This loop repeatedly processes the last digit of 'n' and then removes it
   * until 'n' becomes 0.
   * Pre-condition: 'n' is a non-negative integer.
   * Invariant: 'count' accumulates lucky digits; 'n' reduces with each iteration.
   */
  while (n != 0) {
    // Check if the last digit of 'n' is a lucky digit (4 or 7).
    if (n % 10 == 4 || n % 10 == 7) {
      count++; // Increment the count of lucky digits.
    }
    n = n / 10; // Remove the last digit from 'n'.
  }
  /**
   * @brief Block Logic: Determines if the count of lucky digits is itself lucky.
   *
   * Pre-condition: 'count' holds the total count of lucky digits in the input number.
   * Invariant: Prints "YES" if 'count' is 4 or 7, "NO" otherwise.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl; // The number is "nearly lucky".
  } else {
    cout << "NO" << endl;  // The number is not "nearly lucky".
  }
}
