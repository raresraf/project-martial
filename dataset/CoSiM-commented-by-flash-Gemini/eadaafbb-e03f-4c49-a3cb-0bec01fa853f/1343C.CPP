/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem from Codeforces (Problem 1343C).
 * @details The goal is to find the maximum possible sum of a subsequence where the
 * signs of its elements alternate. The strategy is to greedily pick the largest
 * number from each consecutive block of numbers that share the same sign.
 *
 * @algorithm
 * 1. For each test case, read an array of numbers.
 * 2. Iterate through the array, identifying segments of consecutive numbers with the same sign.
 * 3. For each such segment, find the maximum element.
 * 4. Add this maximum element to a running total sum.
 * 5. Skip the outer loop's index to the end of the processed segment.
 * 6. Continue until the entire array has been processed.
 * 7. Print the total sum.
 *
 * @time_complexity O(N) for each test case, as each element is visited a constant number of times.
 * @space_complexity O(N) to store the input array.
 */
#include <iostream>
#include <vector>
using namespace std;

/**
 * @brief Checks the sign of a given number.
 * @param x The number to check.
 * @return -1 if the number is negative, 1 if it is non-negative.
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
     * @block_logic Main loop to iterate through the array and process blocks of same-signed numbers.
     */
    for (int i = 0; i < n; i++) {
      int j = i;
      long long maxE = (long long)-2e9; // Initialize with a very small number.

      /**
       * @block_logic Inner loop to find the end of the current block of same-signed
       * numbers and find the maximum element within it.
       * @invariant `j` scans forward as long as `arr[j]` has the same sign as `arr[i]`.
       * @post `maxE` holds the largest value in the block `[i, j-1]`.
       */
      while (j < n) {
        if (check(arr[j]) == check(arr[i])) {
          maxE = (maxE < arr[j]) ? arr[j] : maxE;
        } else {
          break;
        }
        j++;
      }

      // Add the maximum element from the found block to the total sum.
      sum = sum + maxE;

      // Move the outer loop index past the block we just processed.
      i = j - 1;
    }
    cout << sum << endl;
  }
}