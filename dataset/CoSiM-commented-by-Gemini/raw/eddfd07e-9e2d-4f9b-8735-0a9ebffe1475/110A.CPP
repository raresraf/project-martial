/**
 * @file 110A.CPP
 * @brief Determines if a number is a "nearly lucky number".
 *
 * A nearly lucky number is defined as a number that has a "lucky" number of
 * lucky digits. A lucky digit is either 4 or 7. This program counts the
 * number of lucky digits in the input number and then checks if that count
 * is itself equal to 4 or 7.
 *
 * @b Algorithm: The program first iterates through each digit of the input number,
 *    counting how many of them are 4 or 7. It then checks if this final count
 *    is equal to 4 or 7.
 * @b Time_Complexity: O(log10(N)), where N is the input number, as the program
 *    processes one digit at a time.
 * @b Space_Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  long long int input, temp;
  cin >> input;
  // The 'flag' variable is declared but never used.
  int flag = 0;
  long long int count = 0;
  temp = input;

  // Block Logic: Count the number of lucky digits (4 or 7) in the input number.
  while (temp) {
    int r = temp % 10;
    if (r == 7 || r == 4)
      count++;
    temp /= 10;
  }

  // Pre-condition: Check if the count of lucky digits is itself a lucky number.
  if (count == 7 || count == 4) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}