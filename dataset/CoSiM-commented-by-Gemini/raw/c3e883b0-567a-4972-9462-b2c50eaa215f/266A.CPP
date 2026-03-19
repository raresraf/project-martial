/**
 * @file 266A.CPP
 * @brief A functionally correct but flawed solution for the "Stones on the Table" problem (Codeforces 266A).
 * @details This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. The algorithm implemented,
 * while producing the correct output, is based on a flawed premise and contains bugs.
 *
 * Algorithm: Indirect Group Counting
 * The code counts the number of times adjacent stones have *different* colors. It then
 * subtracts this count from the total number of stones, 'n'. This indirectly calculates
 * the number of stones to remove (`n - number_of_groups`). While the formula is
 * arithmetically sound, the implementation has serious issues.
 *
 * BUGS AND ISSUES:
 * 1. Out-of-Bounds Access: The loop `for (int j = 0; j < n; j++)` accesses `cl[j + 1]`.
 *    When `j` reaches `n - 1`, this becomes an access to `cl[n]`, which is outside
 *    the array's bounds. The program's success on test cases is likely due to
 *    coincidence or the specific memory layout during testing.
 * 2. Variable-Length Array (VLA): The use of `char cl[n];` is a non-standard C++
 *    extension and is not guaranteed to be supported by all compilers.
 *
 * Time Complexity: O(N), for the two separate loops over the input.
 * Space Complexity: O(N), to store the stones in the character array.
 */

#include <iostream>
using namespace std;

int main() {

  int n, count = 0;
  cin >> n;
  
  // cl: A Variable-Length Array (VLA) to store the stone colors.
  // This is a non-standard C++ extension.
  char cl[n];

  // Read the stone colors into the array.
  for (int i = 0; i < n; i++) {
    cin >> cl[i];
  }
  
  /**
   * Block Logic: Count the number of "color groups" by counting transitions.
   * WARNING: This loop has an out-of-bounds memory access vulnerability.
   * When j = n - 1, it attempts to read cl[n].
   */
  for (int j = 0; j < n; j++) {
    // Increment count every time a stone is different from the next one.
    if (cl[j] != cl[j + 1]) {
      count++;
    }
  }
  
  // The logic 'n - count' coincidentally calculates the correct number of removals.
  // 'count' effectively becomes the number of color groups.
  // Total removals = Total stones - Number of groups.
  cout << n - count << endl;

  return 0;
}