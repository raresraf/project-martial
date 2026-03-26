/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 * @description This program simulates the changes in a school queue consisting of
 *              boys ('B') and girls ('G'). In one second, any boy standing
 *              immediately in front of a girl will swap places with her. This is
 *              repeated for 't' seconds.
 *
 * Algorithm: The simulation is performed over 't' discrete time steps. A key
 *            feature of this implementation is the inner loop's direction.
 * 1. An outer loop runs for 't' iterations, one for each second.
 * 2. An inner loop scans the queue **backwards**, from right to left.
 * 3. When it finds a `(B, G)` pair (a boy at `j-1`, a girl at `j`), it swaps them.
 * 4. The loop counter `j` is then decremented again. This, combined with the
 *    backward iteration, ensures that each boy moves at most once per time step,
 *    correctly modeling the simultaneous nature of the swaps.
 *
 * Time Complexity: O(N * T), where N is the length of the queue and T is the time.
 * Space Complexity: O(N) to store the queue in a character array.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace std;

// --- Common competitive programming macros ---
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
#define mod 1000000007
#define s(a) scanf("%d", &a)
#define sll(a) scanf("%lld", &a)
#define ss(a) scanf("%s", a)
#define ll long long
#define mp make_pair
#define pb push_back
#define fi first
#define se second

// Global character array to store the queue.
char a[55];

int main() {
  int n, t; // n: queue length, t: time in seconds

  s(n), s(t);
  scanf("%s", a);

  /**
   * Block Logic (Outer Loop): Simulates the passage of 't' seconds.
   */
  for (int i = 0; i < t; ++i) {
    /**
     * Block Logic (Inner Loop): Scans the queue backwards to perform swaps for one second.
     * Iterating from right to left is a clean way to handle the simultaneous swaps.
     */
    for (int j = n - 1; j > 0; --j) {
      // Pre-condition: Find a girl to the right of a boy.
      if (a[j] == 'G' && a[j - 1] == 'B') {
        // Swap the boy and the girl.
        swap(a[j], a[j - 1]);
        // Decrement j to move past the swapped pair. This ensures that the boy,
        // now at position j, isn't considered for another swap in this time step.
        j--;
      }
    }
  }

  printf("%s
", a);

  return 0;
}
