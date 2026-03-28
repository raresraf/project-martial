/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem (Codeforces 1343C).
 *
 * @details The problem requires finding the maximum possible sum by partitioning the
 * array into contiguous blocks of elements with the same sign. From each block,
 * the largest element is chosen, and the final answer is the sum of these elements.
 *
 * @algorithm The code iterates through the array, identifying these contiguous
 * blocks. For each block, it finds the maximum element. This maximum element is
 * added to a running sum. The outer loop index is then advanced to the end of the
 * processed block to ensure each element is considered only once.
 *
 * @time_complexity O(N) per test case, where N is the number of elements.
 * @space_complexity O(N) per test case to store the array elements.
 */
#include <iostream>
#include <vector>
using namespace std;

/**
 * @brief Helper function to determine the sign of a number.
 * @param x The number to check.
 * @return 1 for positive or zero, -1 for negative.
 */
int check(long long x) {
  if (x < 0)
    return -1;
  else
    return 1;
}
int main() {
  int t;
  cin >> t;
  // Main loop for test cases.
  while (t--) {
    int n;
    cin >> n;
    long long value;
    long long sum = 0;
    vector<long long> arr;
    for (int i = 0; i < n; i++) {
      cin >> value;
      arr.push_back(value);
    }

    /**
     * @block_logic This nested loop structure processes the array in contiguous blocks
     * of same-signed numbers. `i` marks the start of a block.
     */
    for (int i = 0; i < n; i++) {
      int j = i;
      long long maxE = (long long)-1e9; // Initialize with a sufficiently small number.

      // Inner loop to find the end of the current block and the max element within it.
      while (j < n) {
        if (check(arr[j]) == check(arr[i])) {
          maxE = (maxE < arr[j]) ? arr[j] : maxE;
        } else {
          break; // Sign changed, end of block.
        }
        j++;
      }
      sum = sum + maxE; // Add the maximum element of the block to the total sum.
      i = j - 1; // Advance the outer loop index past the processed block.
    }
    cout << sum << endl;
  }
}