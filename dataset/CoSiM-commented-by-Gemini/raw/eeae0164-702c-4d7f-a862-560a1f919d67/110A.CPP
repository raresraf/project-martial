/**
 * @file 110A.CPP
 * @brief Determines if a number is "nearly lucky".
 *
 * A number is considered "nearly lucky" if the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (specifically 4 or 7). This program counts the
 * occurrences of '4' and '7' in a given number and checks if this count is 4 or 7.
 *
 * @b Algorithm: The program iteratively extracts the last digit of the input number.
 *    It checks if the digit is a '4' or a '7' and increments a counter if it is.
 *    After processing all digits, it checks if the final count is equal to 4 or 7.
 *
 * @b Time_Complexity: O(log10(N)), where N is the input number, because the
 *    number of iterations is proportional to the number of digits in N.
 * @b Space_Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <iostream>
using namespace std;

int main() {
  long long x; // The input number.
  cin >> x;
  int c = 0; // Counter for the lucky digits.
  
  /**
   * @brief This loop iterates through each digit of the input number.
   * It extracts the last digit, checks if it's a 4 or 7, and then removes the digit.
   */
  while (x) {
    int d = x % 10; // Get the last digit.
    x = x / 10;     // Remove the last digit.
    if (d == 4 || d == 7)
      c++;
  }
  
  /**
   * @brief Check if the count of lucky digits is itself a lucky number (4 or 7).
   */
  if (c == 4 || c == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
