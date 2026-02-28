/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 339A - "Helpful Maths").
 *
 * This program sorts an arithmetic expression consisting of numbers 1, 2, and 3
 * separated by '+' signs, arranging them in non-decreasing order.
 *
 * Algorithm: Parse, Sort, and Reconstruct.
 * It extracts the numbers (digits '1', '2', '3') from the input string, stores them
 * in a vector of integers, sorts this vector, and then reconstructs the output
 * string by joining the sorted numbers with '+' signs.
 *
 * Time Complexity: O(L + N log N), where L is the length of the input string and
 * N is the count of numbers (approximately L/2). This simplifies to O(L log L)
 * in the worst case.
 * Space Complexity: O(L) to store the input string and the vector of extracted numbers.
 */
#include <algorithm> // Required for std::sort
#include <cmath>     // Not strictly used in this solution, but included in bits/stdc++.h
#include <iostream>  // Required for input/output operations
#include <string>    // Required for std::string
#include <vector>    // Required for std::vector

using namespace std;

/**
 * @brief Converts a character digit to its integer equivalent.
 *
 * This function takes a character (expected to be '1', '2', or '3')
 * and converts it into its corresponding integer value by subtracting
 * the ASCII value of '0' (or 48).
 *
 * @param a The character digit to convert.
 * @return int The integer value of the character.
 */
int conver(char a) {
  return a - 48; // Functional Utility: ASCII conversion from character digit to integer.
}

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads an arithmetic expression, extracts and sorts the numbers,
 * then prints the sorted expression.
 */
int main() {
  string a; // Variable to store the input sum string.
  // Vector to store the extracted integer numbers from the input string.
  vector<int> num_char2;
  // Pre-condition: Read the input sum string 'a'.
  cin >> a;
  // Block Logic: Iterate through the input string, extracting only the digits.
  // Numbers are at even indices (0, 2, 4, ...).
  // Invariant: 'i' advances by 2 to skip '+' characters.
  for (int i = 0; i < a.length(); i = i + 2) {
    num_char2.push_back(conver(a[i])); // Convert char digit to int and add to vector.
  }
  // Functional Utility: Sort the extracted numbers in non-decreasing order.
  sort(num_char2.begin(), num_char2.end());
  // Block Logic: Iterate through the sorted numbers to reconstruct and print the string.
  // Invariant: 'i' is an iterator pointing to elements in num_char2.
  for (vector<int>::iterator i = num_char2.begin(); i != num_char2.end(); i++) {
    // If it's the last element, print it without a '+'.
    if (i == num_char2.end() - 1) {
      cout << *i;
    } else {
      // Otherwise, print the number followed by a '+'.
      cout << *i << "+";
    }
  }
  cout << endl; // Add a newline at the end of the output.
  return 0;
}