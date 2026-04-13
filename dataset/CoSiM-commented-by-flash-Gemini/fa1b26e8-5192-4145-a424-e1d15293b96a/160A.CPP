/**
 * @file 160A.CPP
 * @brief This program finds the minimum number of coins to pick from a given
 *        set such that the sum of the picked coins is strictly greater than
 *        the sum of the remaining coins.
 * Algorithm: Greedy approach with sorting. The largest coins are picked first.
 * Time Complexity: O(N log N) due to `std::sort`, where N is the number of coins.
 *                  The subsequent greedy selection loop is O(N).
 * Space Complexity: O(N) for storing coin values in an array.
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
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  // N: number of coins.
  // a[100]: array to store coin values (max 100 coins).
  // s: total sum of all coins.
  // nb: count of coins picked.
  // st: sum of picked coins.
  int N, a[100], i, s = 0, nb = 0, st = 0;
  // Reads the number of coins.
  cin >> N;
  /**
   * Block Logic: Reads N coin values into array 'a' and calculates their total sum 's'.
   * Invariant: After each iteration, one coin value is read and added to the total sum.
   */
  for (i = 0; i < N; i++) {
    cin >> a[i];
    s += a[i];
  }
  // Sorts the coin values in non-decreasing order.
  sort(a, a + N);
  // Starts picking coins from the largest value (end of the sorted array).
  i = N - 1;
  /**
   * Block Logic: Greedily picks coins (starting from the largest) until the sum of
   *              picked coins ('st') is strictly greater than the sum of remaining coins ('s - st').
   * Invariant: In each iteration, the largest available coin is added to 'st', and 'nb' is incremented.
   *            The loop continues as long as 'st' is not strictly greater than 's - st'.
   */
  while (i >= 0 && st <= s - st) {
    st += a[i]; // Add the current largest coin to the sum of picked coins.
    i--;        // Move to the next largest coin.
    nb++;       // Increment the count of picked coins.
  }
  // Prints the minimum number of coins required.
  cout << nb << endl;
  return 0;
}
