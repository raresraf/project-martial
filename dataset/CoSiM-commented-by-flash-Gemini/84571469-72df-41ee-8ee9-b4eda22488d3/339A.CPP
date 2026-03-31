/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (Problem 339A).
 * This program reorders a sum of numbers (e.g., "1+3+2") such that the numbers
 * are in non-decreasing order.
 *
 * Algorithm: Parse the numbers from the input string, sort them, and then
 * reconstruct the string with the sorted numbers.
 * Time Complexity: O(L log N), where L is the length of the input string and N is the count of numbers.
 *                  The dominant factor is sorting N numbers.
 * Space Complexity: O(N) to store the parsed numbers.
 */

#include <algorithm> // Required for std::sort
#include <iostream>  // Required for standard input/output streams (cin, cout)
#include <stdlib.h>  // Required for general utilities (e.g., exit), though not strictly used in this snippet.
#include <vector>    // Required for std::vector container

using namespace std;

int main() {
  /**
   * @brief Main function to reorder a sum of numbers.
   * Reads an input string containing numbers separated by '+', parses these numbers,
   * sorts them in non-decreasing order, and then prints the reordered sum.
   */
  string a, b; // @brief 'a' stores the input string, 'b' stores the reordered output string.
  
  // Block Logic: Read the input string containing numbers and '+' signs.
  cin >> a;
  
  vector<int> sum; // @brief Vector to store the parsed numbers.
  
  // Block Logic: Parse numbers from the input string.
  // Iterate through the string, extracting characters at even indices, which represent numbers.
  // This assumes the input format is strictly "N+N+N...".
  for (int i = 0; i < a.length(); i += 2) {
    sum.insert(sum.end(), a[i]);
  }
  
  // Block Logic: Sort the extracted numbers in non-decreasing order.
  // This is the core step for reordering the sum.
  sort(sum.begin(), sum.end());
  
  // Block Logic: Reconstruct the output string with sorted numbers.
  // Iterate through the sorted numbers, append each to 'b', and add a '+' sign.
  for (vector<int>::iterator i = sum.begin(); i != sum.end(); i++) {
    b += char(*i);
    b.append("+");
  }
  
  // Inline: Remove the trailing '+' character from the reconstructed string.
  b.pop_back();
  
  // Block Logic: Print the final reordered sum string.
  cout << b;
}
