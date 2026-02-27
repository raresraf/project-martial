/**
 * @file 339A.CPP
 * @brief Solution for competitive programming problem: "Helpful Maths".
 *
 * @details This program takes a string representing a sum (e.g., "1+3+2")
 *          and reorders the numbers in non-decreasing order, maintaining
 *          the '+' signs, then prints the resulting sum.
 *          Algorithm: Parses numbers from the input string, stores them in a vector,
 *                     sorts the vector, and then reconstructs the output string.
 *          Time Complexity: O(L log K), where L is the length of the input string
 *                           (for parsing) and K is the number of integers in the string
 *                           (for sorting the vector of numbers).
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s; // Declare string 's' to store the input sum.
  cin >> s;  // Read the input sum string.
  
  vector<int> v; // Declare a vector to store the numbers extracted from the input string.
  
  // Block Logic: Parse the input string, extracting numbers and ignoring '+' signs.
  // Invariant: 'v' contains all numbers from 's' in their original order.
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '+')
      continue; // Skip '+' characters.
    else
      v.push_back(s[i] - '0'); // Convert character digit to integer and add to vector.
  }
  
  // Sort the extracted numbers in non-decreasing order.
  sort(v.begin(), v.end());
  
  // Block Logic: Reconstruct and print the sum string with sorted numbers.
  // Invariant: The numbers are printed in non-decreasing order, separated by '+'.
  for (int i = 0; i < v.size(); i++) {
    // Precondition: If it's not the last number, append '+' after it.
    if (i != v.size() - 1)
      cout << v[i] << "+";
    else
      cout << v[i] << endl; // If it's the last number, append newline instead of '+'.
  }
  
  return 0; // Indicate successful program termination.
}
