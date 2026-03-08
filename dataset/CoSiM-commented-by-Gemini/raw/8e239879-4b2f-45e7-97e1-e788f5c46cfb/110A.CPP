/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem (e.g., Codeforces 110A).
 * @details A "nearly lucky number" is a number where the count of its "lucky digits" (4 and 7)
 * is itself a lucky number. This program first counts the occurrences of digits 4 and 7
 * in the input number, and then checks if that count is 4 or 7.
 *
 * @algorithm Digit-by-digit iteration and counting.
 * @complexity
 * - Time: O(log10(n)), where 'n' is the input number. The number of loop iterations
 *   is equal to the number of digits in 'n'.
 * - Space: O(1), using a fixed amount of memory regardless of the input size.
 */
#include <iostream>

using namespace std;

int main() {
  // The input number, using long long to accommodate up to 18 digits.
  long long n;
  cin >> n;

  // Counter for the number of lucky digits (4 or 7) found.
  int count = 0;
  
  /**
   * @block Lucky Digit Counting
   * @logic Iterates through each digit of the input number 'n'.
   */
  while (n != 0) {
    // Check if the last digit is a 4 or a 7.
    if (n % 10 == 4 || n % 10 == 7) {
      count += 1;
    }
    // Remove the last digit to process the next one.
    n /= 10;
  }

  /**
   * @block Final Check
   * @logic Determines if the count of lucky digits is itself a lucky number (4 or 7).
   * The problem constraints imply we only need to check for 4 and 7.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
