/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem (Codeforces 339A).
 *
 * @details
 * This program reads a string representing a sum of single-digit numbers (1, 2, or 3),
 * for example, "3+2+1". It then rearranges the numbers and prints them in
 * non-decreasing order, separated by plus signs. For instance, "3+2+1" becomes "1+2+3".
 *
 * The implementation uses a counting sort algorithm, which is highly efficient for
 * sorting elements from a small, fixed range.
 *
 * Algorithm: Counting Sort
 * 1. Count the frequency of each number ('1', '2', '3') in the input string.
 * 2. Reconstruct the string by printing the numbers in ascending order ('1's, then '2's, then '3's),
 *    interspersed with '+' signs.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(1), as the counting array has a fixed size.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string inp;
  cin >> inp;
  int a[4] = {0};
  int len = inp.length();
  int plus_count = 0;

  /**
   * @brief Counting phase of the sorting algorithm.
   * @details This loop parses the input string, stepping by 2 to read only the digits
   * and skip the '+' signs. It counts the occurrences of each digit (1, 2, 3)
   * and stores the frequencies in the array `a`.
   */
  for (int i = 0; i < len; i += 2) {
    // Convert character '1', '2', '3' to integer 1, 2, 3.
    int j = (int)inp[i] - 48;
    a[j]++;
    plus_count++;
  }

  /**
   * @brief Reconstruction phase of the sorting algorithm.
   * @details This section iterates through the possible numbers in ascending order (1, 2, 3).
   * For each number, it prints it as many times as it appeared in the input,
   * followed by a '+' sign, until all numbers have been printed.
   */
  for (int i = 1; i < 4; i++) {
    // Invariant: All numbers less than `i` have been printed.
    while (a[i] > 0) {
      cout << i;
      plus_count--;
      if (plus_count > 0)
        cout << '+';
      a[i]--;
    }
  }
}