/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces (Problem 266A).
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * @description
 * There are n stones of three possible colors (R, G, B) arranged in a row.
 * The program calculates the minimum number of stones to remove so that no two
 * adjacent stones have the same color.
 *
 * @algorithm
 * The algorithm iterates through the stones from left to right. It compares each
 * stone with the one immediately next to it. If two adjacent stones have the same
 * color, one of them must be removed. The program maintains a counter which is
 * incremented for each such pair found. This count represents the minimum number
 * of removals required.
 *
 * @time_complexity O(N), where N is the number of stones, as it involves a single
 *                  pass through the list of stones.
 * @space_complexity O(N) for storing the colors of the stones in a character array.
 *                   Given the constraint N <= 50, this is a small fixed buffer.
 */
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  // n: the total number of stones.
  int n;
  // count: the number of stones to remove.
  int count = 0;
  // str: character array to store the colors of the stones.
  char str[50];

  // Read the number of stones and their colors.
  cin >> n;
  cin >> str;

  /**
   * @brief Block Logic: Count adjacent stones of the same color.
   * This loop iterates through the stones and compares each stone with its
   * immediate neighbor to the right. If they match, the removal counter is
   * incremented.
   * Note: The loop condition `i < n` might lead to accessing `str[n]` when `i = n-1`,
   * which is out-of-bounds for a string of length n. However, it often works
   * in practice in competitive programming environments due to null terminators
   * or memory layout, but the correct, safe bound would be `i < n - 1`.
   */
  for (int i = 0; i < n; i++) {
    if (str[i] == str[i + 1])
      count++;
    else
      continue;
  }
  
  // Output the minimum number of stones to remove.
  cout << count << endl;

  return 0;
}
