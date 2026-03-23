/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A, "Nearly Lucky Number".
 * @details The problem defines a "lucky number" as a positive integer whose decimal
 * representation contains only the digits 4 and 7. A "nearly lucky number" is a
 * number where the count of lucky digits ('4' or '7') within it is a lucky number.
 * This program checks if a given number is a nearly lucky number.
 *
 * @algorithm Digit counting and checking
 * @complexity
 * Time: O(L), where L is the number of digits in the input number, because it
 *       iterates through the string representation of the number once.
 * Space: O(L) to store the input number as a string.
 */
#include <bits/stdc++.h> // A non-standard header for including all standard libraries
using namespace std;

int main() {
  string n;
  cin >> n; // Read the number as a string to handle potentially large inputs.

  int len = n.length();
  int lucky_digit_counter = 0; // Counter for the lucky digits '4' and '7'.

  /**
   * @brief Loop to count the occurrences of lucky digits.
   */
  for (int i = 0; i < len; i++) {
    if (n[i] == '4' || n[i] == '7')
      lucky_digit_counter++;
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * The only single-digit lucky numbers are 4 and 7. The problem constraints
   * imply we only need to check for these.
   */
  if (lucky_digit_counter == 4 || lucky_digit_counter == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  // A final newline is good practice, though not always required by online judges.
  // cout << endl; 
  
  return 0; // The return is implicit in modern C++, but explicit is good practice.
}