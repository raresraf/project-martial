/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" programming contest problem.
 * @details A number is "nearly lucky" if the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number. This implementation reads the input as an integer and
 * processes its digits arithmetically.
 *
 * Algorithm:
 * 1. Read the input number as an integer `n`. Note: this limits the input size
 *    compared to reading it as a string.
 * 2. Initialize a counter for lucky digits to 0.
 * 3. Use a loop to extract each digit from the number:
 *    a. Get the last digit using the modulo operator (`n % 10`).
 *    b. Check if the digit is 4 or 7.
 *    c. If it is, increment the counter.
 *    d. Remove the last digit by integer division (`n / 10`).
 *    e. Repeat until the number becomes 0.
 * 4. After the loop, check if the final count is a lucky number (i.e., 4 or 7,
 *    as these are the only relevant lucky numbers for a realistic digit count).
 * 5. Print "YES" if it is, "NO" otherwise.
 *
 * Time Complexity: O(log10(N)), where N is the input number, as the loop runs once per digit.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n;
  int count = 0;
  
  /**
   * @block
   * @description This loop extracts and processes each digit of the input number
   * to count the occurrences of lucky digits '4' and '7'.
   */
  while (n != 0) {
    // Check if the last digit is a lucky digit.
    if (n % 10 == 4 || n % 10 == 7) {
      count++;
    }
    // Remove the last digit from the number.
    n = n / 10;
  }
  
  /**
   * @block
   * @description Checks if the total count of lucky digits is itself a lucky number.
   * For the constraints of the problem, only counts of 4 and 7 are possible lucky numbers.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}