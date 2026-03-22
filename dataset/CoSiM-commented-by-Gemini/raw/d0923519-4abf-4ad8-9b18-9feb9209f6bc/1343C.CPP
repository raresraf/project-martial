/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem with a greedy approach.
 *
 * The goal is to find a subsequence from a given sequence of numbers that has the
 * maximum possible sum, under the constraint that the signs of its elements must alternate.
 *
 * @b Algorithm: Greedy Strategy
 *   This solution partitions the array into contiguous blocks of numbers with the same sign.
 *   To get the maximum sum, it selects the single largest value from each block. A
 *   pre-processing step stores the sign of each number in a boolean vector to simplify
 *   block detection.
 *
 * @b Time_Complexity: O(N) per test case. Although there are nested loops, each
 *   element is processed a constant number of times because the outer loop's index
 *   is advanced to the end of the inner block.
 * @b Space_Complexity: O(N) per test case to store the input sequence and the sign vector.
 */
#include <iostream>
#include <vector>
using namespace std;
typedef long long int ll;

// Global vectors to store the sequence and the sign of each element.
vector<int> seq;
vector<bool> pos; // true if positive, false if negative/zero.
int N;

/**
 * @brief Calculates the max sum of an alternating subsequence using the global vectors.
 * @return The maximum possible sum.
 */
ll calc() {
  int j;
  ll ret = 0;
  int mx;
  
  // Iterate through the sequence, processing one same-sign block at a time.
  for (int i = 0; i < N; i++) {
    j = i + 1;
    mx = seq[i];
    
    // Find the end of the current block of numbers with the same sign,
    // while also finding the maximum element in this block.
    while (j < N && pos[j] == pos[i]) {
      mx = max(mx, seq[j]);
      j++;
    }
    
    // Add the largest element of the block to the total sum.
    ret += mx;
    
    // Jump the index to the end of the processed block for the next iteration.
    i = j - 1;
  }
  return ret;
}

int main() {
  // Fast I/O for competitive programming.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  
  int T;
  cin >> T;
  while (T--) {
    cin >> N;
    // Resize global vectors for the current test case.
    seq.resize(N);
    pos.resize(N);
    
    // Read the sequence and populate the boolean sign vector.
    for (int n = 0; n < N; n++) {
      cin >> seq[n];
      pos[n] = (seq[n] > 0);
    }
    
    cout << calc() << "\n";
  }
  return 0;
}
