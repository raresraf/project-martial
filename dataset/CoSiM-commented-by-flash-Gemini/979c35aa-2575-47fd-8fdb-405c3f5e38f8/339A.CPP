/**
 * @file 339A.CPP
 * @brief This program reorders the terms of an arithmetic sum containing only digits '1', '2', and '3'
 * such that the terms are in non-decreasing order. The input is a string representing the sum,
 * and the output is the reordered sum string.
 *
 * Algorithm: Counting Sort / Frequency Counting.
 * 1. Count the occurrences of each digit ('1', '2', '3') in the input string.
 * 2. Print the digits back in ascending order, separated by '+'.
 * Time Complexity: O(N) where N is the length of the input string, due to a single pass for counting and another for printing.
 * Space Complexity: O(1) as only a fixed number of counters are used.
 */

#include <iostream>
#include <string.h>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads the arithmetic sum string, counts the digits, and prints the sorted sum.
 * @pre Input string `s` contains only digits '1', '2', '3' and '+' signs.
 * @post Prints the reordered sum string where terms are in non-decreasing order.
 * @return 0 upon successful execution.
 */
int main() {
  string s;
  // Functional Utility: 'one', 'two', and 'three' store the counts of digits '1', '2', and '3' respectively.
  int one = 0, two = 0, three = 0;
  cin >> s;

  // Block Logic: Iterates through the input string to count the occurrences of each digit.
  // Invariant: At each iteration, `one`, `two`, and `three` hold the correct counts
  //            for the processed part of the string.
  for (int i = 0; i < s.length(); i++) {
    // Block Logic: Increments the respective counter if the current character is a digit '1', '2', or '3'.
    if (s.at(i) == '1')
      one++;
    if (s.at(i) == '2')
      two++;
    if (s.at(i) == '3')
      three++;
  }

  // Block Logic: Prints the '1's, '2's, and '3's in ascending order,
  // separated by '+' signs, ensuring no trailing '+'.

  // Block Logic: Prints all '1's.
  // Pre-condition: 'one' contains the total count of '1's.
  if (one != 0) {
    for (int i = 1; i < one; i++)
      cout << "1+";
    // Block Logic: Determines if a '+' is needed after the last '1'.
    // Pre-condition: All preceding '1's have been printed.
    if (two == 0 && three == 0)
      cout << 1;
    else
      cout << "1+";
  }

  // Block Logic: Prints all '2's.
  // Pre-condition: 'two' contains the total count of '2's.
  if (two != 0) {
    for (int i = 1; i < two; i++)
      cout << "2+";
    // Block Logic: Determines if a '+' is needed after the last '2'.
    // Pre-condition: All preceding '2's have been printed.
    if (three == 0)
      cout << 2;
    else
      cout << "2+";
  }

  // Block Logic: Prints all '3's.
  // Pre-condition: 'three' contains the total count of '3's.
  if (three != 0) {
    for (int i = 1; i < three; i++)
      cout << "3+";
    cout << 3;
  }

  return 0;
}
