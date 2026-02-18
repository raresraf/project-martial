/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (110A) from Codeforces.
 * @details A number is "nearly lucky" if the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (i.e., the count is 4 or 7).
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution counts the occurrences of lucky digits and then checks if the count is lucky.
 * 1. Read the input number.
 * 2. Initialize a counter for lucky digits to 0.
 * 3. Use a loop to process the number digit by digit:
 *    a. Get the last digit using the modulo operator (`% 10`).
 *    b. If the digit is 4 or 7, increment the counter.
 *    c. Remove the last digit using integer division (`/ 10`).
 *    d. Repeat until the number becomes 0.
 * 4. After the loop, check if the final count of lucky digits is 4 or 7.
 * 5. Print "YES" if it is, "NO" otherwise.
 *
 * @complexity
 * Time Complexity: O(log10(N)), where N is the input number. The loop runs once for each digit in the number.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

int main() {

  long long int a;
  cin >> a;

  int ndigits = 0; // Counter for the number of lucky digits ('4' or '7').
  /**
   * @block
   * @brief Iterates through each digit of the input number to count lucky digits.
   * @invariant `ndigits` holds the count of lucky digits from the processed part of the number.
   * @pre `a` is the input number.
   * @post `ndigits` holds the total count of '4's and '7's in the original number `a`.
   */
  while (a != 0) {
    if (a % 10 == 4 || a % 10 == 7)
      ndigits++;
    a /= 10;
  }

  /**
   * @block
   * @brief Checks if the count of lucky digits is itself a lucky number.
   */
  if (ndigits == 4 || ndigits == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
