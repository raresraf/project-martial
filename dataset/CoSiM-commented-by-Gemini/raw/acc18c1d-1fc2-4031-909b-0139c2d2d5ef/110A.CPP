/**
 * @file 110A.CPP
 * @brief Determines if a number is "nearly lucky".
 *
 * A number is considered "nearly lucky" if the count of its "lucky digits"
 * (4 and 7) is a lucky number itself (i.e., the count is 4 or 7).
 * This program reads an integer, counts the occurrences of digits 4 and 7,
 * and then checks if this count is 4 or 7.
 *
 * @see https://codeforces.com/problemset/problem/110/A
 */
#include <iostream>

using namespace std;

int main() {
  long long n;
  cin >> n;
  int count = 0;
  /**
   * @brief Counts the lucky digits.
   *
   * This loop iterates through each digit of the input number `n`. It checks
   * if the last digit is 4 or 7 and increments the `count` accordingly.
   */
  while (n != 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      count += 1;
    }
    n /= 10;
  }
  /**
   * @brief Checks if the count is a lucky number.
   *
   * After counting the lucky digits, this conditional statement checks if the
   * `count` itself is a lucky number (4 or 7).
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}