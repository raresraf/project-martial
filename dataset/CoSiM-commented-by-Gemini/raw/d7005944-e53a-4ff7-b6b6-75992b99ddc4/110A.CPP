/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A, "Nearly Lucky Number".
 * @details This program checks if a number is "nearly lucky". A number is nearly lucky
 * if the count of its "lucky digits" (4 and 7) is itself a lucky number.
 * This implementation uses integer arithmetic to count the lucky digits.
 *
 * @algorithm Digit counting via modulo arithmetic.
 * @complexity
 * Time: O(log10(N)), where N is the input number, because it iterates through each digit.
 * Space: O(1).
 */
// codeforces110a.cpp
#include <iostream>
using namespace std;

/**
 * @brief Checks if a given number is a "lucky number" (composed only of 4s and 7s).
 * @param n The number to check.
 * @return True if n is a lucky number, false otherwise.
 */
bool islucky(long long int n) {
  // A count of 0 lucky digits is not a lucky number itself.
  if (n == 0) {
    return false;
  }
  // The 't' variable acts as a flag. Its initial value is not used due to the logic below.
  // A clearer approach would be to initialize a boolean flag to true.
  bool t = true; 
  while (n > 0) {
    // Check if the last digit is not 4 and not 7.
    if (n % 10 != 4 && n % 10 != 7) {
      t = false; // If not, the number is not lucky.
      break;
    }
    n = n / 10; // Move to the next digit.
  }
  return t;
}

int main(int argc, char const *argv[]) {
  long long int n, b;
  long long int d = 0; // Counter for the number of lucky digits.

  cin >> n;
  b = n; // Use a copy of n to preserve the original value.

  /**
   * @brief Loop to count the number of lucky digits (4 or 7) in the input number.
   */
  while (b > 0) {
    if (b % 10 == 4 || b % 10 == 7)
      d++;
    b /= 10;
  }

  /**
   * @brief Check if the count of lucky digits ('d') is itself a lucky number.
   */
  if (islucky(d))
    cout << "YES";
  else
    cout << "NO";
    
  return 0; // Indicate successful execution.
}