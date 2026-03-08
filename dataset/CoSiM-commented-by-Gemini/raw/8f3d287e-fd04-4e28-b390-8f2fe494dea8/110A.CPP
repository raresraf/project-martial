/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem (e.g., Codeforces 110A).
 * @details A number is "nearly lucky" if the count of its "lucky digits" (4 and 7)
 * is itself a lucky number. This program counts the occurrences of 4 and 7 in the
 * input number's decimal representation and then checks if that count is 4 or 7.
 *
 * @algorithm Digit-by-digit iteration and counting.
 * @complexity
 * - Time: O(log10(n)), where 'n' is the input number, as the loop runs once for
 *   each digit in 'n'.
 * - Space: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h> // A non-standard header that includes most standard library headers.
using namespace std;

int main() {
  // Use long long to handle large input numbers (up to 18 digits).
  long long num;
  cin >> num;

  // 'count' will store the number of lucky digits found.
  int count = 0;
  
  /**
   * @block Lucky Digit Counting
   * @logic Iteratively processes each digit of the input number from right to left.
   */
  while (num != 0) {
    // Check if the last digit is a 4 or a 7.
    if (num % 10 == 4 || num % 10 == 7) {
      count += 1;
    }
    // Remove the last digit.
    num /= 10;
  }
  
  /**
   * @block Final "Nearly Lucky" Check
   * @logic Determines if the total count of lucky digits is itself a lucky number.
   * Based on problem constraints, the count will be small, so only 4 and 7 need to be checked.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
