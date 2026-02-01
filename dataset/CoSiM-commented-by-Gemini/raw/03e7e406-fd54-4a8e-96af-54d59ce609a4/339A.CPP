/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces contest 339.
 *
 * @details
 * This program takes a string representing a sum of single-digit numbers (1, 2, or 3),
 * sorts the numbers in non-decreasing order, and prints the new sum. For example,
 * an input of "3+2+1" should result in an output of "1+2+3".
 *
 * Algorithm:
 * 1. Read the input string which contains digits separated by '+' signs.
 * 2. Extract all the digits from the string and store them in a vector of integers.
 *    The digits are located at even indices (0, 2, 4, ...).
 * 3. Sort the vector of digits in non-decreasing order.
 * 4. Construct and print the output string by iterating through the sorted digits
 *    and inserting a '+' sign between them.
 *
 * Time Complexity: O(N log N) where N is the number of digits in the input string.
 * This is dominated by the sorting operation.
 * Space Complexity: O(N) to store the digits in a vector.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;

  // Read the entire line containing the sum.
  cin >> s;

  vector<int> v;

  // Block Logic: Extracts the numbers from the string.
  // The numbers are at even positions (0, 2, 4, ...).
  // 's[i] - 48' converts the character digit to an integer.
  for (int i = 0; i < s.length(); i = i + 2) {

    int k = s[i] - 48; // Convert char to int (ASCII '0' is 48)

    v.push_back(k);
  }

  // Functional Utility: Sorts the extracted numbers in non-decreasing order.
  sort(v.begin(), v.end());

  int j = 0;

  // Block Logic: Constructs and prints the sorted sum string.
  // It iterates through the required length of the output string.
  // Even positions get a number from the sorted vector, and odd positions get a '+'.
  for (int i = 0; i < s.length(); i++) {

    if (i % 2 == 0) {
      // Place a number at even positions.
      cout << v[j];

      j++;

    }

    else {
      // Place a '+' at odd positions.
      cout << '+';
    }
  }
}