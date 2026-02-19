/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem.
 * @details The program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * @algorithm The solution uses a single-pass greedy approach. It iterates through the
 * sequence of stones, comparing each stone to its immediate neighbor. If a stone is
 * identical to the next one, it is counted as one to be removed. The final count
 * represents the total minimum number of removals required.
 *
 * @time_complexity O(N), where N is the number of stones, as the algorithm involves a single
 * pass through the stone sequence.
 * @space_complexity O(N), for storing the sequence of stones in an array.
 */
#include <iostream>

using namespace std;
int main() {
  int n;
  // `numberDiff` is a counter for the minimum number of stones to remove.
  // The name is misleading, as it counts adjacent stones that are the SAME color.
  int numberDiff;
  cin >> n;
  numberDiff = 0;

  char stones[55];
  // Block Logic: Read the sequence of N stone colors into an array.
  for (int i = 0; i < n; i++) {
    cin >> stones[i];
  }

  /**
   * Block Logic: Iterate through the stones to count adjacent pairs of the same color.
   * @note The loop condition `i < n` can lead to an out-of-bounds read at `stones[i+1]`
   * when `i` is `n-1`. The intended logic is likely `i < n - 1`.
   */
  for (int i = 0; i < n; i++) {
    // Pre-condition: Check if the current stone and the next are identical.
    if (stones[i] == stones[i + 1]) {
      numberDiff++;
    } else {
      // No action needed if colors are different.
    }
  }
  cout << numberDiff;
}
