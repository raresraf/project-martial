/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A, "Helpful Maths".
 * @details This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and reorders it into a non-decreasing sum (e.g., "1+2+3").
 * This implementation works by parsing the numbers into an integer array,
 * sorting the array, and then printing the result.
 *
 * @section
 *
 * Algorithm:
 * The process is divided into three main steps:
 * 1.  **Parsing:** The input string is iterated over. Characters at even indices,
 *     which correspond to the digits, are extracted. A `stringstream` is used
 *     to convert each digit character into an actual integer, which is then
 *     stored in an integer array.
 * 2.  **Sorting:** The C++ standard library's `std::sort` function is called on
 *     the integer array to sort the numbers in non-decreasing order.
 * 3.  **Output Generation:** The program iterates through the sorted array,
 *     printing each number and re-inserting the '+' symbol between them to
 *     reconstruct the final, sorted sum string.
 *
 * - Time Complexity: O(N log N), dominated by the `std::sort` algorithm, where N
 *   is the number of digits in the sum.
 * - Space Complexity: O(N) to store the integer array.
 */
#include <algorithm>
#include <iostream>
#include <sstream>
using namespace std;

int main() {
  string input, temp;
  cin >> input;

  int len = input.length() / 2 + 1;
  int data[len], k = 0;

  /**
   * Block Logic: Parses the input string to extract the digits.
   * It iterates through the string, picking characters at even indices (the digits)
   * and converting them from char to int using a stringstream before storing
   * them in the 'data' array.
   */
  for (int i = 0; i < input.length(); i++)
    if (i % 2 == 0) {
      temp = input.at(i);
      stringstream ss(temp);
      ss >> data[k++];
      temp = "";
    }

  int n = sizeof(data) / sizeof(data[0]);
  // Sorts the array of numbers using the standard library's introspective sort.
  sort(data, data + n);

  /**
   * Block Logic: Prints the sorted numbers, re-inserting '+' symbols to
   * format the output as a sum.
   */
  for (int i = 0; i < len; i++)
    if (i == len - 1)
      cout << data[i] << endl;
    else
      cout << data[i] << '+';

  return 0;
}
