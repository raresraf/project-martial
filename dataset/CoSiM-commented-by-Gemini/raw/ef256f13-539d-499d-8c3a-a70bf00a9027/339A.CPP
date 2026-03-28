/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers into non-decreasing order.
 *
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2"), extracts the number characters, sorts them, and then
 * prints the rearranged sum (e.g., "1+2+3").
 *
 * @b Algorithm:
 * 1. The input string is parsed, and every second character (the digits)
 *    is stored in a vector of characters.
 * 2. The vector of characters is sorted. Since ASCII values of digits are ordered,
 *    this correctly sorts the numbers.
 * 3. The sorted characters are printed, separated by '+'.
 *
 * @b Time_Complexity: O(N log N), where N is the number of summands, dominated
 *    by the sorting algorithm.
 * @b Space_Complexity: O(N) to store the summand characters in a vector.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  string ch;
  int i, j;
  cin >> ch;
  vector<char> numbers;
  
  /**
   * @brief This loop extracts the digit characters from the input string.
   * It iterates with a step of 2 to skip the '+' signs.
   */
  for (i = 0; i < ch.length(); i = i + 2) {
    numbers.push_back(ch[i]);
  }
  
  // Sort the vector of characters.
  sort(numbers.begin(), numbers.begin() + numbers.size());
  
  /**
   * @brief This loop prints the sorted sum.
   * It prints each digit character followed by a plus, except for the last one.
   */
  for (i = 0; i < numbers.size() - 1; i++) {
    cout << numbers[i] << "+";
  }
  // Print the last digit.
  cout << numbers[i] << endl;
  
  return 0;
}
