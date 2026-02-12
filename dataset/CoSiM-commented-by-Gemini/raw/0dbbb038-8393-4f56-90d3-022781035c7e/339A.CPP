/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem using a counting sort approach.
 * @details The program sorts the numbers in a string representing a sum (e.g., "3+1+2")
 * into non-decreasing order (e.g., "1+2+3"). This implementation uses a counting sort,
 * which is highly efficient for a small, fixed range of input values (1, 2, and 3).
 *
 * @algorithm Counting Sort
 * 1. Read the input string.
 * 2. Iterate through the string, counting the occurrences of '1', '2', and '3'.
 * 3. Print the sorted sum by first printing all the '1's, then all the '2's,
 *    and finally all the '3's, separated by '+' signs.
 *
 * @complexity
 * Time complexity: O(L), where L is the length of the string, as it involves a few linear passes. This is more efficient than a comparison-based sort.
 * Space complexity: O(1), as it only uses a few counter variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  // ck: a boolean flag to control the printing of the '+' operator.
  bool ck = false;
  cin >> a;
  
  int len = a.length();
  // Counters for the number of 1s, 2s, and 3s.
  int num1 = 0, num2 = 0, num3 = 0;
  
  /**
   * @brief First pass: Count the occurrences of each digit.
   */
  for (int i = 0; i < len; i++) {
    if (a[i] == '1') {
      num1++;
    }
    if (a[i] == '2') {
      num2++;
    }
    if (a[i] == '3') {
      num3++;
    }
  }
  
  /**
   * @brief Second pass: Construct the output string from the counts.
   */
  // Print all the '1's.
  for (int i = 0; i < num1; i++) {
    if (ck == false) {
      ck = true; // The first number has been printed.
    } else {
      cout << '+';
    }
    cout << 1;
  }
  // Print all the '2's.
  for (int i = 0; i < num2; i++) {
    if (ck == false) {
      ck = true;
    } else {
      cout << '+';
    }
    cout << 2;
  }
  // Print all the '3's.
  for (int i = 0; i < num3; i++) {
    if (ck == false) {
      ck = true;
    } else {
      cout << '+';
    }
    cout << 3;
  }
}