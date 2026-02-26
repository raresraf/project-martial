/**
 * @file 266A.CPP
 * @brief An attempt to solve the "Stones on the Table" problem.
 * @details The problem is to find the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. This requires counting
 * pairs of identical adjacent stones.
 *
 * @algorithm (Intended)
 * 1. Read the number of stones, `n`, and the sequence of colors.
 * 2. Iterate through the sequence from the first to the second-to-last stone.
 * 3. In each iteration, compare the current stone's color with the next one.
 * 4. If they are the same, increment a counter representing stones to be removed.
 * 5. Print the final counter.
 *
 * @note
 * The implementation below is **flawed and incorrect**. The input reading logic
 * is fundamentally broken, and the subsequent processing operates on improperly
 * read data, leading to an incorrect result. The comments describe the
 * apparent intent of the code rather than its actual behavior.
 *
 * @complexity (Intended)
 * Time Complexity: O(N), for a single pass through the stones.
 * Space Complexity: O(N), to store the sequence of stone colors.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  char a[101];
  int i, n, j = 0, count = 0;
  scanf("%d", &n);

  /**
   * @block
   * @brief Incorrect input reading loop.
   * @warning This loop is flawed. Using `scanf` with "%s" inside a loop like
   * this is designed to read strings, not individual characters. It will try
   * to read a full string into `&a[i]`, causing a buffer overflow and not
   * populating the array with the intended sequence of colors. The correct way
   * would be to read the entire string at once, e.g., `cin >> a;`.
   */
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]); // This does not work as intended.
  }

  // This line overwrites the first element, corrupting any data that might have been read.
  a[j] = 'a';

  /**
   * @block
   * @brief Main logic loop (operating on invalid data).
   * @details The apparent intent is to compare adjacent elements. As `j` is
   * incremented in every iteration, `a[j]` is always the character just before
   * `a[i]` (i.e., `a[i-1]`). The goal is to count `i` where `a[i] == a[i-1]`.
   * However, since the input was not read correctly, this comparison is
   * performed on undefined data.
   */
  for (i = 1; i <= n; i++) {
    if (a[i] == a[j]) {
      count++;
      j++;
    } else {
      j++;
    }
  }

  // The output is incorrect due to the preceding logical errors.
  cout << count << endl;
}