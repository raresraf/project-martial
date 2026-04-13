/**
 * @file 339A.CPP
 * @brief This program processes a string containing a sum of digits '1', '2', and '3'
 *        (e.g., "1+2+3+1"). It counts the occurrences of each digit and then
 *        prints them in non-decreasing order, separated by '+' signs.
 * Algorithm: Counting Sort (implicit) - digits are counted and then printed in order.
 * Time Complexity: O(N) where N is the length of the input string, due to a single pass
 *                  for counting and a single pass for printing.
 * Space Complexity: O(1) for storing digit counts and auxiliary variables.
 */
// ACCEPTED
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  // Reads the input string, e.g., "1+2+3+1".
  cin >> s;
  // total: Stores the total count of digits in the input string.
  // subtotal: Tracks the number of digits printed so far to correctly place '+' signs.
  int total = 0, subtotal = 0;
  // c1, c2, c3: Counters for digits '1', '2', and '3' respectively.
  int c1 = 0, c2 = 0, c3 = 0;
  /**
   * Block Logic: Iterates through the input string to count occurrences of digits '1', '2', and '3'.
   * Invariant: After each iteration, if a digit is encountered, its respective counter is incremented,
   *            and the total count of digits is updated.
   */
  for (char a : s) {
    if (a == '1') {
      c1++;
      total++;
    }
    if (a == '2') {
      c2++;
      total++;
    }
    if (a == '3') {
      c3++;
      total++;
    }
  }

  /**
   * Block Logic: Prints all '1's found in the input string, separated by '+'.
   * Pre-condition: 'c1' contains the count of digit '1'.
   * Invariant: Decrements 'c1' and increments 'subtotal' after each '1' is printed,
   *            adding a '+' unless it's the last digit overall.
   */
  while (c1 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "1"
           << "+";
    else
      cout << "1";

    c1--;
  }
  /**
   * Block Logic: Prints all '2's found in the input string, separated by '+'.
   * Pre-condition: 'c2' contains the count of digit '2'.
   * Invariant: Decrements 'c2' and increments 'subtotal' after each '2' is printed,
   *            adding a '+' unless it's the last digit overall.
   */
  while (c2 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "2"
           << "+";
    else
      cout << "2";

    c2--;
  }
  /**
   * Block Logic: Prints all '3's found in the input string, separated by '+'.
   * Pre-condition: 'c3' contains the count of digit '3'.
   * Invariant: Decrements 'c3' and increments 'subtotal' after each '3' is printed,
   *            adding a '+' unless it's the last digit overall.
   */
  while (c3 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "3"
           << "+";
    else
      cout << "3";

    c3--;
  }
}
