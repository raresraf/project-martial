/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem.
 * @see https://codeforces.com/problemset/problem/160/A
 *
 * This program determines the minimum number of coins to take so that the sum
 * of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm:
 * 1. Read the number of coins (n) and the value of each coin.
 * 2. Calculate the total sum of all coins.
 * 3. Sort the coins in descending order.
 * 4. Iterate through the sorted coins, adding each coin to a "twin" sum and
 *    subtracting it from the total sum.
 * 5. Count the number of coins taken.
 * 6. Stop when the twin sum becomes strictly greater than the remaining sum.
 * 7. Print the number of coins taken.
 *
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) to store the coin values.
 */
/*-----rk_98-----*/
#include <bits/stdc++.h>
#define fast()                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define ll long long
#define mod 1000000007
#define eps 1e-9
#define ff first
#define ss second
#define pb push_back
#define endl "\n"
using namespace std;
int main() {
  // fast();
  ll n;
  cin >> n;
  ll sum = 0, arr[n];
  for (ll i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  // Sorts the coins in descending order to greedily pick the largest ones first.
  sort(arr, arr + n, greater<ll>());
  ll twin = 0, coin = 0;
  // Invariant: This loop iterates through the sorted coins, accumulating the
  // sum for one twin and stopping when it exceeds the sum of the remaining coins.
  for (ll i = 0; i < n; i++) {
    twin += arr[i];
    sum -= arr[i];
    coin++;
    if (twin > sum)
      break;
  }
  cout << coin << endl;
}