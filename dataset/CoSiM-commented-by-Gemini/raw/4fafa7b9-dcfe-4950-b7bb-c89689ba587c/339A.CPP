/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem by sorting integers.
 * @details The program takes a string representing a sum (e.g., "3+2+1") and
 * rearranges the numbers into non-decreasing order (e.g., "1+2+3"). This
 * version explicitly converts the digit characters to integers before sorting.
 *
 * @algorithm
 * 1. Read the input string containing the sum.
 * 2. Create an empty integer vector.
 * 3. Iterate through the input string. If a character is '1', '2', or '3', push
 *    the corresponding integer value into the vector. This filters out the '+' signs.
 * 4. Sort the integer vector using `std::sort`.
 * 5. Reconstruct the output string by iterating through the sorted vector and
 *    printing each number, separated by '+'.
 *
 * @complexity
 * Time Complexity: O(N log N), where N is the number of digits in the sum.
 * The performance is dominated by the sorting operation.
 * Space Complexity: O(N) to store the integer representation of the numbers.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  vector<int> numbers; // Vector to store the integer values of the digits.
  string expression;   // Input string, e.g., "1+3+2".
  cin >> expression;

  /**
   * @block
   * @brief Parse the string to extract digits as integers.
   */
  for (int i = 0; i < expression.length(); i++) {
    if (expression[i] == '1')
      numbers.push_back(1);
    else if (expression[i] == '2')
      numbers.push_back(2);
    else if (expression[i] == '3')
      numbers.push_back(3);
    // The '+' characters are implicitly ignored.
  }

  // Sort the vector of numbers in non-decreasing order.
  sort(numbers.begin(), numbers.end());

  /**
   * @block
   * @brief Print the sorted numbers in sum format.
   * @note The logic here is slightly convoluted. It iterates up to the
   * second-to-last element, printing a '+' after each, and then prints the
   * last element separately. A more common approach is to print the '+'
   * *before* each element except the first.
   */
  int last_printed_index = 0;
  for (auto it = numbers.begin(); it != (numbers.end() - 1); ++it) {
    cout << *it << "+";
    last_printed_index++;
  }

  // Print the last element without a trailing '+'.
  if (!numbers.empty()) {
      cout << numbers[last_printed_index];
  }
  
  cout << endl;
  return 0;
}