/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem.
 *
 * @details This program calculates the minimum number of stones to remove from a row
 * on a table so that no two adjacent stones have the same color. The colors are
 * given as a sequence of characters.
 *
 * @section
 * Algorithm:
 * The approach is to iterate through the stones from left to right and count
 * how many times a stone is followed immediately by another of the same color.
 * Each such pair requires one stone to be removed.
 *
 * Time Complexity: O(N), where N is the number of stones, due to the linear scan.
 * Space Complexity: O(N), to store the sequence of stone colors.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int count = 0;

  char s[100];

  /**
   * @brief Reads the sequence of n stone colors into a character array.
   */
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  /**
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   * Invariant: 'count' holds the number of removals needed for the stones processed so far.
   * @note The loop termination condition depends on `strlen(s)`, which assumes a null-terminated
   * string. It compares each stone `s[i]` with its neighbor `s[i+1]`.
   */
  for (int i = 0; i < strlen(s); i++) {
    /**
     * @brief Checks if a stone and the one next to it are the same color.
     * Logic: If two adjacent stones are identical, one must be removed. The 'count'
     * is incremented to track the total number of required removals.
     */
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  cout << count << endl;
}
