/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem, a classic greedy algorithm puzzle.
 *
 * This program determines the minimum number of coins a person must take
 * from a set so that the sum of their coins is strictly greater than the sum
 * of the remaining coins.
 *
 * @algorithm
 * 1. Read the total number of coins and the value of each coin.
 * 2. Calculate the total sum of all coin values.
 * 3. Sort the coins in non-decreasing order.
 * 4. Greedily iterate through the coins from largest to smallest, accumulating their sum.
 * 5. Stop when the accumulated sum is greater than half of the total sum.
 * 6. The number of coins taken up to that point is the answer.
 *
 * @time_complexity O(N log N) due to the sorting of the coin values, where N is the number of coins.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  /**
   * @brief Optimizes I/O operations by decoupling C++ streams from C standard streams.
   * This is a common practice in competitive programming for faster execution.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int N, a[100], i, s = 0, nb = 0, st = 0;
  cin >> N;

  // Read the coin values and calculate their total sum.
  for (i = 0; i < N; i++) {
    cin >> a[i];
    s += a[i];
  }

  // Sort the coins in ascending order to facilitate the greedy approach.
  sort(a, a + N);

  i = N - 1;
  /**
   * @brief Greedily take the largest coins until the sum is greater than the remaining sum.
   *
   * @invariant `st` holds the sum of coins taken by the twin.
   * @invariant `s - st` holds the sum of the remaining coins.
   * The loop continues as long as the twin's sum is not strictly greater than the other's.
   */
  while (i >= 0 && st <= s - st) {
    st += a[i];
    i--;
    nb++;
  }

  // Output the minimum number of coins required.
  cout << nb << endl;
  return 0;
}
