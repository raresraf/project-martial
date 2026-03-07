// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" competitive programming problem (Codeforces 110A).
 * A number is considered "lucky" if it contains only the digits 4 and 7.
 * A number is considered "nearly lucky" if the count of lucky digits (4s and 7s)
 * within the number is itself a lucky number.
 * Algorithm: Digit extraction, counting, and then re-checking the count's digits.
 * Time Complexity: O(log10(N)), where N is the input number (as it iterates through digits).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  long long n; // Declares 'n' to store the input number (using long long for potentially large numbers).
  // Reads the input number.
  cin >> n;
  
  long long count = 0; // Initializes 'count' to store the number of lucky digits (4s and 7s) in 'n'.
  /**
   * Block Logic: Iterates through the digits of 'n' to count occurrences of 4 and 7.
   * Invariant: At the start of each iteration, 'n' is the remaining part of the original number,
   * and 'count' holds the lucky digits found in the processed part.
   */
  while (n != 0) {
    int x = n % 10; // Extracts the last digit of 'n'.
    n /= 10;        // Removes the last digit from 'n'.
    // Conditional Logic: Checks if the extracted digit 'x' is either 4 or 7.
    if (x == 4 || x == 7) {
      count++; // Increments the lucky digit count.
    }
  }
  
  // Conditional Logic: If no lucky digits were found in the original number.
  if (count == 0) {
    cout << "NO" << endl; // Prints "NO" as it cannot be a nearly lucky number.
    return 0;             // Terminates the program.
  }
  
  bool flag = true; // Flag to indicate if the 'count' itself is a lucky number.
  /**
   * Block Logic: Iterates through the digits of 'count' to check if it contains only 4s and 7s.
   * Invariant: At the start of each iteration, 'count' is the remaining part of the lucky digit count,
   * and 'flag' reflects the "luckiness" of the processed part.
   */
  while (count != 0) {
    int x = count % 10; // Extracts the last digit of 'count'.
    count /= 10;        // Removes the last digit from 'count'.
    // Conditional Logic: Checks if the extracted digit 'x' is neither 4 nor 7.
    if (x != 4 && x != 7) {
      flag = false;       // Sets flag to false, as 'count' is not a lucky number.
      cout << "NO" << endl; // Prints "NO".
      break;              // Exits loop early.
    }
  }
  
  // Final Conditional Logic: If 'flag' is still true after checking all digits of 'count'.
  if (flag)
    cout << "YES" << endl; // Prints "YES", indicating it's a nearly lucky number.
  
  return 0; // Indicates successful program execution.
}