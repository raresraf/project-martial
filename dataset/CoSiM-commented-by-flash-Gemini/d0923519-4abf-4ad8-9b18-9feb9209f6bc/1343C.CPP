/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem from Codeforces (1343C).
 * @details The goal is to find a subsequence with the maximum possible sum, where
 * the signs of adjacent elements alternate.
 *
 * @algorithm
 * This solution uses a greedy strategy. For each contiguous block of numbers
 * with the same sign (all positive or all negative), it selects the largest
 * number from that block to be part of the final subsequence.
 * 1. For each test case, read the input numbers.
 * 2. Pre-process the input by storing the numbers in one vector (`seq`) and
 *    their signs (`true` for positive, `false` otherwise) in a parallel
 *    boolean vector (`pos`).
 * 3. In a `calc` function, iterate through the sequence.
 * 4. For each element, start scanning forward to find the end of the block
 *    of elements with the same sign, using the pre-calculated `pos` vector.
 * 5. While scanning the block, keep track of the maximum element found.
 * 6. Add this maximum element to the total sum.
 * 7. Advance the main loop's index past the just-processed block.
 * 8. Return the total sum.
 *
 * @time_complexity O(N) for each test case, as each element is visited a constant number of times.
 * @space_complexity O(N) to store the sequence and the sign vector.
 */
#include <iostream>
#include <vector>
using namespace std;
typedef long long int ll;

// Global vectors to store the sequence and the sign of each element.
vector<int> seq;
vector<bool> pos;
int N;

/**
 * @brief Calculates the maximum sum of an alternating subsequence.
 * @return The calculated maximum sum.
 */
ll calc() {
  int j;
  ll ret = 0;
  int mx;
  /**
   * @block_logic Iterates through the sequence, processing one block of same-signed numbers at a time.
   */
  for (int i = 0; i < N; i++) {
    j = i + 1;
    mx = seq[i];
    /**
     * @block_logic Inner loop finds the end of the current block and the max element within it.
     * @invariant `j` advances as long as `pos[j]` matches `pos[i]`.
     * @post `mx` holds the largest value in the block `[i, j-1]`.
     */
    while (j < N && pos[j] == pos[i]) {
      mx = max(mx, seq[j]);
      j++;
    }
    ret += mx; // Add the block's maximum element to the result.
    i = j - 1; // Skip the outer loop to the end of the processed block.
  }
  return ret;
}

int main() {
  // Fast I/O
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  int T;
  cin >> T;
  while (T--) {
    cin >> N;
    seq.resize(N);
    pos.resize(N);
    /**
     * @block_logic Read input and pre-calculate the sign of each number.
     */
    for (int n = 0; n < N; n++) {
      cin >> seq[n];
      pos[n] = (seq[n] > 0);
    }
    cout << calc() << "\n";
  }
  return 0;
}
