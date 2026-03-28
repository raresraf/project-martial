/**
 * @file 266B.CPP
 * @brief Simulates a queue rearrangement process with a specific swapping logic.
 *
 * This program simulates a queue of characters. Over a given number of time
 * steps, it iterates through the queue and whenever it finds a 'G', it swaps
 * it with the character immediately preceding it.
 *
 * @b Algorithm: The simulation uses a nested loop structure. The outer loop
 *    progresses time. The inner loop sweeps through the queue. When a 'G' is
 *    found (and it's not at the first position), it's swapped with the element
 *    at `i-1`. The loop index `i` is then incremented to continue the scan from
 *    the position after the swap.
 *
 * @b Time_Complexity: O(time * num), where 'time' is the number of time steps
 *    and 'num' is the length of the queue.
 * @b Space_Complexity: O(num) to store the queue string.
 */
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

// Competitive programming macros for brevity.
#define mp make_pair
#define pb push_back
#define f first
#define s second

using namespace std;

// Typedefs for common data structures.
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> coord;
typedef vector<int> veci;
typedef vector<coord> coords;
const int mod = 1000000007;
const int mod2 = 998244353;

/**
 * @brief Unused function to check if a number is prime.
 */
bool isprime(ull n) {
  if (n % 2 == 0) {
    return n == 2;
  }
  if (n % 3 == 0) {
    return n == 3;
  }
  ull k = sqrt(n);
  for (int i = 5; i < k; i += 2) {
    if (n % k == 0) {
      return false;
    }
  }
  return n != 1;
}

int main() {
  ios_base::sync_with_stdio(false);
  string line;
  int num, time;
  cin >> num >> time >> line;

  /**
   * @brief Outer loop for time steps.
   */
  while (time--) {
    /**
     * @brief Inner loop to scan the queue. When a 'G' is found, it's
     * swapped with the character before it.
     */
    for (int i = 0; i < num; ++i) {
      if (line[i] == 'G' && i) {
        swap(line[i], line[i - 1]);
        // Increment 'i' to avoid re-processing the swapped character.
        ++i;
      }
    }
  }
  cout << line << '\n';
}