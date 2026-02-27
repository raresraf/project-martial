/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces.
 * @details A "lucky number" is a positive integer whose decimal representation
 * contains only the digits 4 and 7. A number is "nearly lucky" if the count
 * of lucky digits in it is a lucky number. This program checks if a given
 * number is nearly lucky.
 *
 * @algorithm
 * 1. Read the input number as a string to easily iterate over its digits.
 * 2. Initialize a counter to zero.
 * 3. Iterate through each character of the string. If the character is '4' or
 *    '7', increment the counter.
 * 4. After counting, check if the counter's value is itself a lucky number.
 *    Given the constraints on the input number's length (up to 18 digits),
 *    the maximum possible count is 18. The only lucky numbers less than or
 *    equal to 18 are 4 and 7.
 * 5. Therefore, if the count is 4 or 7, the number is nearly lucky. Print "YES".
 *    Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * due to the single pass through the string.
 * Space Complexity: O(L) to store the number as a string.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief A generic function to convert a number to a string.
 * @note This function is defined but not used in this particular program.
 */
template <typename T> string NumberToString(T Number) {
  ostringstream ss;
  ss << Number;
  return ss.str();
}

int main() {
  string str; // To store the input number as a string.
  cin >> str;
  
  int count = 0; // Counter for lucky digits ('4' and '7').
  
  /**
   * @block
   * @brief Iterate through the string to count the occurrences of lucky digits.
   */
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '4' || str[i] == '7') {
      count++;
    }
  }
  
  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * For the problem constraints, this simplifies to checking if the count is 4 or 7.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
