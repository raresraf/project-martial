/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem by counting participants based on a cutoff score.
 *
 * This program is a solution to the "Next Round" problem. It determines the number
 * of participants who advance to the next round. The logic reads `n` (total participants)
 * and `k` (the rank establishing the cutoff). It then reads all `n` scores into a
 * vector. The score of the k-th participant is used as the benchmark. The program
 * then counts how many participants have a score greater than or equal to this
 * benchmark score.
 *
 * @note This implementation differs from the standard problem statement which
 *       requires a participant's score to also be positive to advance. This code
 *       does not check for positivity and may produce incorrect results if the
 *       cutoff score is zero or negative.
 *
 * @author f26dcd8b-a356-4c22-b6ea-c7a94c14fb23
 *
 * @algorithm Two-Pass Counting
 * @complexity
 * Time: O(N), for reading the N scores and then iterating through them again.
 * Space: O(N), to store the scores in a vector.
 */

/*
Hey there stalker....
















Like what you see?
*/
#include <algorithm>
#include <bits/stdc++.h>
#include <set>
#include <string>
#include <vector>
using namespace std;
// Macro to optimize C++ stream I/O operations.
#define IOS                                                                    
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);                                                               
  cout.tie(NULL);
// Common competitive programming macros for brevity.
#define pb push_back
#define pii pair<ll, ll>
#define vi vector<ll>
#define all(a) (a).begin(), (a).end()
#define F first
#define S second
#define sz(x) (ll) x.size()
#define rep(i, a, b) for (ll i = a; i < b; i++) // Standard for-loop
#define repr(i, a, b) for (ll i = a; i > b; i--) // Reverse for-loop
#define lbnd lower_bound
#define ubnd upper_bound
#define bs binary_search
#define mp make_pair
#define ll long long int
#define lld long double
#define endl "
"
#define MOD 1000000007 // A common prime modulus

int main() {
  IOS; // Apply I/O optimization.

  // The code is structured to handle multiple test cases, but it's commented out.
  int TESTS = 1;
  // cin >> TESTS;
  while (TESTS--) {
    // `n`: total participants, `k`: cutoff rank.
    ll n, k;
    // `a`: stores the score of the k-th participant.
    // `count`: stores the number of participants who advance.
    ll a, count = 0;
    cin >> n >> k;
    // `v`: a vector to store all participant scores.
    vi v(n);

    // First pass: Read all `n` scores into the vector.
    rep(i, 0, n) { cin >> v[i]; }

    // The score of the k-th participant (k-1 index) is the cutoff score.
    a = v[k - 1];

    /**
     * @brief Second pass: Count participants who meet the criteria.
     * Pre-condition: `v` is populated and `a` holds the cutoff score.
     * Invariant: `count` is incremented for each score that is greater than
     *            or equal to the cutoff score `a`.
     */
    rep(i, 0, n) {
      if (v[i] >= a) {
        count++;
      }
    }
    // Print the final count of advancing participants.
    cout << count << endl;
  }
  return 0;
}