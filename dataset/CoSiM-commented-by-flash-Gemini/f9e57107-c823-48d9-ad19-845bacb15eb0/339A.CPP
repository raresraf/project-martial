/**
 * @file 339A.CPP
 * @brief This program solves the "Helpful Maths" problem. It takes an input string
 * representing a sum of '1', '2', and '3' digits separated by '+' signs (e.g., "3+2+1")
 * and outputs them in non-decreasing order (e.g., "1+2+3").
 * Algorithm: Counting sort. It counts the occurrences of each digit ('1', '2', '3')
 * and then directly prints them to standard output in sorted order, inserting '+'
 * signs as separators.
 * Time Complexity: O(L) where L is the length of the input string, due to multiple passes.
 * Space Complexity: O(1) for digit counters.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
using namespace std;     // Brings all names from the std namespace into the current scope.

int main() {
  // Optimizes C++ standard streams by disabling synchronization with C stdio.
  ios_base::sync_with_stdio(0);
  string s; // Stores the input sum string (e.g., "1+3+2").
  // Reads the input sum string.
  cin >> s;
  int a = 0, b = 0, c = 0; // Counters for digits '1', '2', and '3' respectively.

  /**
   * Block Logic: Iterates through the input string to count the occurrences of each digit ('1', '2', '3').
   * Plus signs '+' are ignored as they are not digits.
   * Invariant: 'a', 'b', 'c' accurately reflect the counts of '1's, '2's, and '3's in the string.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      a++;
    if (s[i] == '2')
      b++;
    if (s[i] == '3')
      c++;
  }

  /**
   * Block Logic: Prints all '1's, separated by '+' signs.
   * Iterates 'a' times, printing '1'. A '+' is printed after each '1' except the last.
   */
  for (int i = 0; i < a; i++) {
    if (i == a - 1) // If it's the last '1'.
      cout << "1";
    else
      cout << "1+";
  }

  /**
   * Block Logic: Prints all '2's, separated by '+' signs, if any '2's exist.
   * If '1's were printed before, a '+' separator is added before printing '2's.
   */
  if (b != 0) { // Only proceed if there are '2's to print.
    if (a != 0) // If any '1's were printed before.
      cout << "+"; // Add a '+' separator.
    for (int i = 0; i < b; i++) {
      if (i == b - 1) // If it's the last '2'.
        cout << "2";
      else
        cout << "2+";
    }
  }

  /**
   * Block Logic: Prints all '3's, separated by '+' signs, if any '3's exist.
   * If any '1's or '2's were printed before, a '+' separator is added before printing '3's.
   */
  if (c != 0) { // Only proceed if there are '3's to print.
    if (a != 0 || b != 0) // If any '1's or '2's were printed before.
      cout << "+"; // Add a '+' separator.
    for (int i = 0; i < c; i++) {
      if (i == c - 1) // If it's the last '3'.
        cout << "3";
      else
        cout << "3+";
    }
  }

  // Standard program exit.
  return 0;
}
