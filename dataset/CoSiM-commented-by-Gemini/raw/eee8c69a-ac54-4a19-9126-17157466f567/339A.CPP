/**
 * @file helpfulmaths.cpp
 * @brief Rearranges a sum of numbers into non-decreasing order.
 *
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2"), extracts the numbers, sorts them, and prints the
 * rearranged sum (e.g., "1+2+3").
 *
 * @b Algorithm:
 * 1. The input string is parsed to extract the integer summands.
 * 2. These integers are stored in a vector.
 * 3. The vector is sorted in non-decreasing order.
 * 4. The sorted numbers are printed, separated by '+'.
 *
 * @b Time_Complexity: O(N log N), where N is the number of summands, dominated
 *    by the sorting algorithm.
 * @b Space_Complexity: O(N) to store the summands in a vector.
 */
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Parses a string of the form "a+b+c..." into a vector of integers.
 * @param str The input string representing the sum.
 * @return A vector of integers containing the summands.
 */
vector<int> parseString(string str) {
  stringstream ss(str);
  int integer;
  char plus;
  vector<int> temp;

  /**
   * @brief Loop to extract integers from the string stream.
   * It reads an integer and a plus sign until the end of the stream.
   */
  while (!ss.eof()) {
    ss >> integer >> plus;
    temp.push_back(integer);
  }

  return temp;
}

int main() {
  string s;
  cin >> s;

  // Parse the input string to get the numbers.
  vector<int> summands = parseString(s);

  // Sort the numbers in non-decreasing order.
  sort(summands.begin(), summands.end());

  int i;

  /**
   * @brief Loop to print the sorted sum.
   * It prints each number followed by a plus, except for the last one.
   */
  for (i = 0; i < summands.size() - 1; i++) {
    cout << summands[i] << "+";
  }
  // Print the last number.
  cout << summands[i] << endl;

  return 0;
}
