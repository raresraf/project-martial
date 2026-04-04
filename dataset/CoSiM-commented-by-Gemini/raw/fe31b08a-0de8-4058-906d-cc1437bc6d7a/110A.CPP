/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from platforms like Codeforces.
 * @details A number is "nearly lucky" if the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (i.e., the count is 4 or 7).
 *
 * Algorithm:
 * 1. Read the input number as a string to easily iterate through its digits.
 * 2. Initialize a counter 'noluckey' to 0.
 * 3. Iterate through each character (digit) of the string:
 *    a. If the digit is '4' or '7', increment 'noluckey'.
 * 4. After counting, check if the final count ('noluckey') is equal to 4 or 7.
 * 5. If it is, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the input number as a string.
 */
#include <iostream>

using namespace std;

int main() {
  int length;
  int noluckey = 0;
  string holder;
  cin >> holder;
  length = holder.length();
  // This VLA is non-standard C++ and is not used in the program's logic.
  char almostlucky[length];
  
  /**
   * @brief Loop through each digit of the input number to count lucky digits.
   * Pre-condition: 'holder' contains the input number as a string. 'noluckey' is 0.
   * Invariant: After each iteration, 'noluckey' holds the count of lucky digits
   * found in the string up to the current position.
   */
  for (int i = 0; i < length; i++) {
    almostlucky[i] = holder.at(i); // This assignment is not used for the final result.
    // Check if the current digit is a lucky digit ('4' or '7').
    if (holder.at(i) == '4' || holder.at(i) == '7') {
      noluckey++;
    }
  }

  // A number is "nearly lucky" if the count of lucky digits is itself a lucky number (4 or 7).
  if (noluckey == 4 || noluckey == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}