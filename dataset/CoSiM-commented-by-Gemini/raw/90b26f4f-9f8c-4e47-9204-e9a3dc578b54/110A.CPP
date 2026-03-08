/**
 * @file 110A.CPP
 * @brief An unnecessarily complex solution for the "Nearly Lucky Number" problem (e.g., Codeforces 110A).
 * @details This program determines if a number is "nearly lucky" by checking if the count
 * of its lucky digits (4 and 7) is itself a lucky number (4 or 7).
 *
 * @note This implementation is inefficient. It uses two passes and O(log n) extra space
 * to store digits in an array, whereas a single-pass O(1) space solution is sufficient.
 *
 * @algorithm Two-pass digit counting via an intermediate array.
 * @complexity
 * - Time: O(log10(n)), where 'n' is the input number, as all operations are proportional
 *   to the number of digits in 'n'.
 * - Space: O(log10(n)) to store the digits in the `arr` array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n;
  cin >> n;
  int j = 0, arr[100]; // Array to hold the digits. `100` is oversized but safe.

  // Get the number of digits by converting the number to a string. This is inefficient.
  int len = to_string(n).length();

  /**
   * @block First Pass: Deconstruct number into an array of digits.
   * @logic This loop extracts each digit from the number and stores it in an array.
   * This entire block is unnecessary for solving the problem.
   */
  while (n != 0) {
    arr[len - j - 1] = n % 10;
    n = n / 10;
    j++;
  }
  
  int i = 0, count = 0;

  /**
   * @block Second Pass: Count lucky digits from the array.
   * @logic This loop iterates through the newly created digit array to count 4s and 7s.
   */
  while (len) {
    if (arr[i] == 4 || arr[i] == 7)
      count++;
    len--;
    i++;
  }

  // Check if the final count of lucky digits is itself a lucky number (4 or 7).
  if (count == 4 || count == 7)
    cout << "YES";
  else
    cout << "NO";
}
