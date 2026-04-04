/**
 * @file 753A.CPP
 * @brief Distributes a given sum 'n' into a sequence of distinct positive integers, prioritizing smallest integers.
 *
 * This program aims to find the maximum number of distinct positive integers (1, 2, 3, ...)
 * that can sum up to a value less than or equal to 'n'. It then outputs this count
 * and one possible distribution of these integers, ensuring the sum of the printed
 * integers equals 'n'. This is a classic greedy problem.
 *
 * Algorithm: Greedy approach. Find the largest 'k' such that the sum 1 + 2 + ... + k <= n.
 *            Then print 1 to (k-1) and for the k-th number, print the remaining sum needed to reach 'n'.
 * Time Complexity: O(sqrt(N)) due to finding 'i' (k) by summing consecutive integers,
 *                  and then O(i) for printing. Since i is approx sqrt(2N), it is O(sqrt(N)).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
// Functional Utility: Defines 'F' as a shorthand for 'first' for accessing elements of a pair.
#define F first
// Functional Utility: Defines 'S' as a shorthand for 'second' for accessing elements of a pair.
#define S second
// Functional Utility: Defines 'PB' as a shorthand for 'push_back' for vector operations.
#define PB push_back
// Functional Utility: Defines 'MP' as a shorthand for 'make_pair' for creating pairs.
#define MP make_pair
// Functional Utility: Defines 'REP' as a shorthand for a for-loop from 'a' to 'b-1'.
#define REP(i, a, b) for (int i = a; i < b; i++)
// Functional Utility: Defines 'll' as a shorthand for 'long long' for convenience.
typedef long long ll;
// Functional Utility: Defines 'vi' as a shorthand for 'vector<int>'.
typedef vector<int> vi;
// Functional Utility: Defines 'pi' as a shorthand for 'pair<int, int>'.
typedef pair<int, int> pi;
// Functional Utility: Defines 'llp' as a commonly used large prime number for modulo operations.
#define llp 1000000007
int n;

int main() {
  /**
   * @brief Main function: Finds the maximum number of distinct positive integers that sum up to 'n'
   *        and prints them.
   * @return 0 if the program executes successfully.
   */
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams for faster input/output.
  ios::sync_with_stdio(false);
  cin >> n; // Read the input integer 'n'.
  int i;    // 'i' will store the largest number of distinct integers.
  int k = 0; // 'k' will store the sum 1 + 2 + ... + i.
  /**
   * @brief Loop to find the largest 'i' such that the sum of integers from 1 to 'i' is less than or equal to 'n'.
   * @pre 'i' starts from 1, 'k' (cumulative sum) starts from 0.
   * @post 'i' will be one greater than the actual maximum number of integers, so it's decremented later.
   * @invariant 'k' accumulates the sum of consecutive integers, 'i' tracks the current integer.
   */
  for (i = 1;; i++) { // Infinite loop, breaks when k > n.
    k = i * (i + 1) / 2; // Calculate sum of 1 to 'i' using arithmetic progression formula.
    /**
     * @brief Break condition: if the current sum 'k' exceeds 'n'.
     */
    if (k > n)
      break;
  }
  i--; // Decrement 'i' to get the actual maximum number of distinct integers.
  cout << i << "\n"; // Output the maximum count of distinct integers.
  /**
   * @brief Loop to print the integers from 1 to (i-1).
   * @pre 'i' holds the maximum count of distinct integers.
   * @post Integers from 1 to (i-1) are printed, each followed by a space.
   */
  REP(j, 0, i - 1) { // REP is a macro for a for loop.
    cout << j + 1 << " ";
  }
  // Functional Utility: Calculate the sum of integers from 1 to (i-1).
  // The remaining value 'n - (sum 1 to i-1)' is the last number to print to make the total sum 'n'.
  cout << (n - (i * (i - 1) / 2)) << "\n";
  return 0;
}
