/**
 * @file 110A.CPP
 * @brief Solution for a competitive programming problem involving "nearly lucky numbers".
 *
 * This program determines if a given number is "nearly lucky". A number is
 * considered nearly lucky if the total count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (either 4 or 7). This implementation processes the
 * input as an integer.
 *
 * @algorithm
 * 1. Read the input number as a long long integer.
 * 2. Initialize a counter to zero.
 * 3. Use a loop with modulo and division operations to extract and examine each
 *    digit of the number from right to left.
 * 4. If a digit is '4' or '7', increment the counter.
 * 5. After the loop terminates (when the number becomes zero), check if the
 *    counter's value is 4 or 7.
 * 6. Print "YES" if it is, "NO" otherwise.
 *
 * @time_complexity O(L), where L is the number of digits in the input number.
 * @space_complexity O(1), as memory usage is constant.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n;
  cin >> n;
  // 'cnt' will store the total count of lucky digits ('4' and '7').
  int cnt = 0;

  // Block Logic: Iterate through each digit of the number using arithmetic.
  // Pre-condition: 'n' is the input number.
  // Invariant: The loop continues until all digits of 'n' have been processed.
  while (n != 0) {
    // Extract the last digit.
    int t = n % 10;
    // Check if the digit is a lucky digit.
    if (t == 4 || t == 7)
      cnt++;
    // Remove the last digit from the number.
    n /= 10;
  }

  // Block Logic: Determine if the count of lucky digits is itself a lucky number.
  // Pre-condition: 'cnt' holds the total number of '4's and '7's.
  if (cnt == 4 || cnt == 7)
    cout << "YES";
  else
    cout << "NO";
}
