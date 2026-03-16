/**
 * @file 546A.CPP
 * @brief Counts the number of elements in a sorted list that satisfy a condition.
 *
 * This program reads `n` pairs of integers, sorts them based on the first element,
 * and then finds how many pairs have their second element greater than or equal to a given value `k`.
 * It is a typical competitive programming task that involves sorting and searching.
 *
 * @algorithm Sorting and Linear Scan
 * @details The program first reads `n` pairs and stores them in a vector.
 * The vector is then sorted in ascending order based on the first element of the pair.
 * A linear scan is performed to find the first pair whose second element is at least `k`.
 * The number of remaining elements in the vector from that point onwards is the answer.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting of the `n` pairs.
 * The subsequent linear scan takes at most O(N) time.
 * Space Complexity: O(N) to store the `n` pairs in a vector.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
typedef long double ld;
#define mod 1000000007
#define inf 1000000000000000000
#define rk 256
#define pb push_back
#define mp make_pair
#define vc vector
#define fs first
#define sec second
#define pq priority_queue
#define lb lower_bound
#define ub upper_bound
#define pll pair<ll, ll>
#define pls pair<ll, string>
#define psl pair<string, ll>
#define plc pair<ll, char>
#define pcl pair<char, ll>
#define pss pair<string, string>
#define all(v) v.begin(), v.end()
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define T                                                                      
  ll t = 0;                                                                    
  cin >> t;                                                                    
  for (ll test = 0; test < t; test++)

int main() {
  // Fast I/O
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  ll n;
  cin >> n;
  vc<pll> v(n);
  // Read n pairs of (start, end) times.
  for (ll a = 0; a < n; a++)
    cin >> v[a].fs >> v[a].sec;
  
  // Sort the pairs based on their first element (start time).
  sort(all(v));
  
  ll k;
  cin >> k;

  /**
   * @brief Find the first element that satisfies the condition.
   *
   * Iterate through the sorted vector to find the first pair where the
   * second element (end time) is greater than or equal to k.
   * The problem is interpreted as finding how many sessions are available
   * for someone who becomes free at time k. The sessions are sorted by start time.
   * The condition v[a].sec >= k means the session 'a' is still ongoing at time 'k'.
   * Since the sessions are sorted by start time, we find the first such session
   * and count it along with all subsequent sessions.
   */
  for (ll a = 0; a < n; a++) {
    if (v[a].sec >= k) {
      cout << n - a << endl;
      break;
    }
  }
  return 0;
}
