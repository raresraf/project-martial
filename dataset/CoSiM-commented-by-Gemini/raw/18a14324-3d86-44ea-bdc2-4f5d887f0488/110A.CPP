/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem.
 * @details This program determines if a given number is "nearly lucky". A number is considered
 * nearly lucky if the count of its 'lucky digits' (which are 4 and 7) is itself a lucky number (i.e., 4 or 7).
 *
 * @section algorithm Algorithm
 * 1. Read the input number `n`.
 * 2. Initialize a counter `ct` to 0. This counter will store the number of lucky digits found.
 * 3. Iterate through each digit of the input number `n`. This is done by repeatedly
 *    taking the number modulo 10 (to get the last digit) and then dividing by 10
 *    (to remove the last digit) until the number becomes 0.
 * 4. In each iteration, check if the current digit is either 4 or 7. If it is, increment the counter `ct`.
 * 5. After the loop has processed all digits, the counter `ct` holds the total count of lucky digits.
 * 6. Finally, check if the value of `ct` is 4 or 7.
 *    - If it is, the number is nearly lucky, and the program prints "YES".
 *    - Otherwise, it prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(log10(n)), as the loop runs once for each digit of the input number `n`.
 * - Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to determine if a number is "nearly lucky".
 * @return 0 on successful execution.
 */
int main() {
  long long n;
  int ct = 0; // Counter for lucky digits (4 or 7).
  
  cin >> n;
  
  // Loop through each digit of the number.
  while (n > 0) {
    // Check if the last digit is a lucky digit.
    if (n % 10 == 4 || n % 10 == 7) {
      ct++;
    }
    // Remove the last digit.
    n /= 10;
  }
  
  // Check if the count of lucky digits is itself a lucky number.
  if (ct == 4 || ct == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
