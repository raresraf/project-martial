/**
 * @file 339A.cpp
 * @brief Rearranges a sum of numbers to be in non-decreasing order using a multiset.
 * @details This program solves the "Helpful Maths" problem from Codeforces. It takes a
 * string representing a sum of numbers (1s, 2s, and 3s), sorts the numbers,
 * and prints them back as a sum in non-decreasing order.
 *
 * @see https://codeforces.com/problemset/problem/339/A
 *
 * Algorithm:
 * 1. The input string (e.g., "3+1+2") is read.
 * 2. A `std::multiset` is used to store the numbers. A multiset is a sorted container,
 *    so inserting elements automatically keeps them in order.
 * 3. The program iterates through the input string, parsing the characters '1', '2', '3'
 *    and inserting them as integers into the multiset.
 * 4. A new string is built by iterating through the sorted multiset. Each number is
 *    appended to the result string, followed by a '+' character.
 * 5. The resulting string (which has a trailing '+') is printed, omitting the last character.
 *
 * Time Complexity: O(N log N), where N is the length of the string. Inserting N/2 elements
 * into a multiset takes O(log K) for each element, where K is the multiset size.
 * Space Complexity: O(N), to store the numbers in the multiset and the result string.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <set> // Explicitly include for multiset
#include <string> // Explicitly include for string

using namespace std;

int main() {
  string s;
  cin >> s;

  // A multiset automatically stores elements in sorted order.
  multiset<int> v;

  /**
   * @brief Parses the input string and inserts numbers into the multiset.
   * This loop iterates through the string and adds the numerical digits to the
   * multiset, which handles the sorting automatically.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      v.insert(1);
    if (s[i] == '2')
      v.insert(2);
    if (s[i] == '3')
      v.insert(3);
  }

  string res = "";
  /**
   * @brief Constructs the output string from the sorted numbers.
   * This loop iterates through the sorted multiset and builds a string like "1+2+3+".
   */
  for (auto it = v.begin(); it != v.end(); it++) {
    res += to_string(*it);
    res += '+';
  }

  // Print the resulting string, excluding the trailing '+'.
  for (int i = 0; i < res.length() - 1; i++) {
    cout << res[i];
  }
  return 0;
}
