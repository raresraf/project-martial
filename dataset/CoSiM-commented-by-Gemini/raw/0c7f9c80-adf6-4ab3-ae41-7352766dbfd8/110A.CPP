/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem.
 * @details A number is defined as "nearly lucky" if the number of lucky digits ('4' or '7')
 * it contains is itself a lucky number. For the scope of this problem, the only lucky
 * number counts that matter are 4 and 7. This program counts the occurrences of '4' and '7'
 * in the input number string and checks if this count is equal to 4 or 7.
 *
 * @algorithm Digit counting and direct comparison.
 * @complexity
 * Time complexity: O(L), where L is the number of digits in the input number, due to a single pass over the string.
 * Space complexity: O(L) to store the input number as a string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // testStr: stores the input number as a string.
  string testStr;
  // counter: to count the number of lucky digits found.
  int counter = 0;

  cin >> testStr;

  /**
   * @brief Loop to count the lucky digits in the input string.
   * @invariant Iterates through each character of the string.
   */
  for (int i = 0; i < testStr.length(); i++) {
    // If the character is a lucky digit, increment the counter.
    if (testStr[i] == '4' || testStr[i] == '7') {
      counter++;
    }
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number (4 or 7).
   * @post Prints "YES" if the condition is met, otherwise "NO".
   */
  if (counter == 4 || counter == 7) {
    cout << "YES" << endl;
  }

  else {
    cout << "NO" << endl;
  }

  return 0;
}