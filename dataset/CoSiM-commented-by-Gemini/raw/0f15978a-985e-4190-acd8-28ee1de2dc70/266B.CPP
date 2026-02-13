/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem (Codeforces 266B).
 * @author competitive-programmer
 *
 * @details This program simulates the process of rearranging a queue of boys ('B')
 * and girls ('G'). In each time step, any boy standing immediately in front of a
 * girl swaps places with her. This is repeated for a specified number of time steps.
 *
 * Algorithm:
 * 1. Read the length of the queue (n), the number of time steps (k), and the
 *    initial arrangement of the queue as a string (s).
 * 2. Loop 'k' times to simulate each time step.
 * 3. Inside the time step loop, iterate through the queue from left to right.
 * 4. If a 'B' is found immediately before a 'G', swap their positions.
 * 5. To ensure that a boy moves at most one position per time step (atomic movement),
 *    a boolean array `okay` is used. When a swap occurs at index `i`, `okay[i+1]`
 *    is marked as `false` to prevent the boy, now at `i+1`, from being part of
 *    another swap in the same time step.
 * 6. After 'k' time steps, print the final arrangement of the queue.
 *
 * Time Complexity: O(k * n), where k is the number of time steps and n is the
 * length of the queue.
 * Space Complexity: O(n) for the `okay` array.
 */
#include <bits/stdc++.h>
using namespace std;

// Macro for a for-loop.
#define rep(i, a, b) for (int i = a; i < b; i++)
typedef long long int ll;
#define pb push_back
#define elif else if
#define mp make_pair
#define F first
#define S second

int main() {
  // n: length of the queue, k: number of time steps.
  ll n, k;
  cin >> n >> k;
  string s;
  cin >> s;
  ll len = s.size();

  /**
   * Block Logic: This outer loop simulates the progression of time for 'k' steps.
   * Invariant: At the beginning of each iteration, 's' represents the state of
   * the queue at the current time.
   */
  while (k--) {
    // `okay` array is used to ensure atomic swaps within a single time step.
    // A boy involved in a swap at index `i` cannot be swapped again from `i+1`
    // in the same time step.
    bool okay[n];
    memset(okay, 1, sizeof(okay));

    /**
     * Block Logic: This inner loop scans the queue for 'BG' pairs to perform swaps.
     */
    rep(i, 0, len - 1) {
      // Pre-condition: The check `okay[i] == true` ensures that the character at `s[i]`
      // was not just moved into this position in the current time step.
      if (okay[i] == true) {
        // A 'B' followed by a 'G' triggers a swap.
        if (s[i] == 'B' && s[i + 1] == 'G') {
          swap(s[i], s[i + 1]);
          // Mark the new position of the boy as ineligible for another swap in this pass.
          okay[i + 1] = false;
        }
      }
    }
  }
  
  // Print the final state of the queue after k seconds.
  cout << s << endl;

  return 0;
}
