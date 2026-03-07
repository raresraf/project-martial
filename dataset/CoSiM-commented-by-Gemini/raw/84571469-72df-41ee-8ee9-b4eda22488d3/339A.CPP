/**
 * @file 339A.CPP
 * @brief Rearranges a string representing a sum of single-digit numbers into non-decreasing order.
 *
 * @details
 * This program takes a string representing a sum of single-digit numbers (e.g., "3+1+2")
 * and reorders the numbers to be in non-decreasing order (e.g., "1+2+3").
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Extract the numeric characters (digits) from the string. The digits are located at even indices.
 * 3. Store the ASCII values of these digits in a vector of integers.
 * 4. Sort the vector. Since the ASCII values of '0' through '9' are consecutive, sorting the ASCII values is equivalent to sorting the digits.
 * 5. Reconstruct the output string by concatenating the sorted digits with '+' signs in between.
 * 6. Remove the trailing '+' from the end of the string.
 * 7. Print the resulting sorted sum string.
 *
 * Time Complexity: O(L log L), where L is the number of summands (approximately half the length of the input string). The complexity is dominated by the sort operation.
 * Space Complexity: O(L) to store the numbers in a vector.
 */
#include <algorithm> // sort
#include <iostream>
#include <stdlib.h> //exit
#include <vector>

using namespace std;

int main() {
  string a, b;
  cin >> a;
  vector<int> sum;
  /**
   * @brief Extracts the digits from the input string.
   * The loop iterates over the string with a step of 2, picking up characters at even indices, which correspond to the numbers.
   * The characters are stored as their ASCII integer values in the vector.
   */
  for (int i = 0; i < a.length(); i += 2) {
    sum.insert(sum.end(), a[i]);
  }
  
  // Sorts the vector of numbers. Since characters are stored as ASCII values,
  // this correctly sorts the digits '1' through '9'.
  sort(sum.begin(), sum.end());

  /**
   * @brief Reconstructs the sorted sum string.
   * Iterates through the sorted vector, converting each integer back to a character
   * and appending it to the result string, followed by a '+'.
   */
  for (vector<int>::iterator i = sum.begin(); i != sum.end(); i++) {
    b += char(*i);
    b.append("+");
  }
  
  // Removes the extraneous '+' from the end of the string.
  b.pop_back();
  cout << b;
}
