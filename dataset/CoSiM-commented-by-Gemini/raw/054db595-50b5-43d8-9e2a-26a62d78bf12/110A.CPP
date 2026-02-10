/**
 * @file 110A.CPP
 * @brief Solution for competitive programming problem "Nearly Lucky Number".
 * @details This program determines if a given number is a "nearly lucky number".
 * A nearly lucky number is defined as a number whose count of "lucky digits"
 * ('4' and '7') is itself a lucky number. The only lucky numbers relevant for
 * the constraints of this problem are 4 and 7.
 *
 * Algorithm:
 * 1. Read the input number as a string to handle large values.
 * 2. Iterate through each character of the string.
 * 3. Count the occurrences of the digits '4' and '7'.
 * 4. Sum these counts to get the total number of lucky digits.
 * 5. Check if this total sum is equal to 4 or 7.
 * 6. If it is, the number is "nearly lucky"; otherwise, it is not.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the input number as a string.
 */
#include <iostream>
using namespace std;
int main() {

  char a[20];
  // Read the input number as a string.
  cin >> a;
  int i, flag, l, c4, c7, c;
  l = 0;
  c4 = 0;
  c7 = 0;
  c = 0;

  // Calculate the length of the input string.
  while (a[l] != '\0')
    l++;

  // Invariant: c4 and c7 store the counts of '4' and '7' digits found so far.
  for (i = 0; i < l; i++) {
    if (a[i] == '4')
      c4++;
    else if (a[i] == '7')
      c7++;
  }

  // A number is "nearly lucky" if the count of its lucky digits ('4' or '7')
  // is itself a lucky number (4 or 7).
  if (c4 + c7 == 4 || c4 + c7 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
