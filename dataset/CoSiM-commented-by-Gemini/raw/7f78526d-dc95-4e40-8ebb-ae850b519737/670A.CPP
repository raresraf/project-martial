/**
 * @file 670A.CPP
 * @brief Calculates the sum of MEX values for two sets derived from an input array.
 *
 * This program solves a problem where an input array of integers is partitioned
 * into two sets:
 * 1. A set 'a' containing all unique elements from the input.
 * 2. A set 'b' containing all elements that appear more than once in the input.
 *
 * It then calculates the MEX (Minimum Excluded non-negative integer) for each set
 * and prints their sum. The program is structured to handle multiple test cases.
 *
 * @algorithm
 * 1. Partition input elements into a set of unique values and a set of duplicates.
 * 2. Calculate the MEX for each set.
 * 3. Sum the two MEX values.
 *
 * @complexity
 * Time: O(N log N) per test case, where N is the number of elements. The dominant
 *       operation is inserting N elements into a std::set, which takes O(log N)
 *       per insertion. The MEX calculation takes O(K) where K is the set size.
 * Space: O(N) in the worst case to store all unique elements in the sets.
 */
#pragma GCC optimize("-O3")
#include <bits/stdc++.h>
using namespace std;

// --- Competitive Programming Boilerplate ---
// Type definitions for convenience
typedef long long int ll;
typedef long long unsigned lluu;
typedef long double ld;
typedef pair<int, int> p32;
typedef pair<ll, ll> p64;
typedef pair<double, double> pdd;
typedef vector<ll> v64;
typedef vector<int> v32;
typedef vector<vector<int>> vv32;
typedef vector<vector<ll>> vv64;
typedef vector<p64> vp64;
typedef vector<p32> vp32;
ll MOD = 998244353;
ll NUM = 1e9 + 7;
// Macros for common loops and operations
#define forn(i, e) for (ll i = 0; i < e; i++)
#define forsn(i, s, e) for (ll i = s; i < e; i++)
#define rforn(i, s) for (ll i = s; i >= 0; i--)
#define rforsn(i, s, e) for (ll i = s; i >= e; i--)
#define ln "
"
#define dbg(x) cout << #x << " = " << x << ln
#define mp make_pair
#define pb push_back
#define ff first
#define ss second
#define INF 1e18
// Macro to disable C++ stream synchronization with C stdio for faster I/O.
#define fast_cin()                                                             
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);                                                               
  cout.tie(NULL)
#define all(x) (x).begin(), (x).end()
#define sz(x) ((ll)(x).size())
#define zer ll(0)
#define printarr(arr, n) forn(i, n) cout << arr[i] << " "
#define input(arr, n) forn(i, n) cin >> arr[i]

/**
 * @brief Calculates the MEX (Minimum Excluded) value of a given set of integers.
 * @param st A constant reference to a std::set<int>. The set must be sorted.
 * @return The smallest non-negative integer not present in the set.
 *
 * For example, for the set {0, 1, 3, 4}, the MEX is 2.
 */
int mex(const set<int> &st) {
  auto it = st.begin();
  int i = 0;
  // Pre-condition: The set is sorted by default.
  // Block Logic: Iterate through the set, comparing each element with the
  // expected sequence of non-negative integers (0, 1, 2, ...).
  for (; i < st.size(), it != st.end(); i++, it++) {
    // If the element in the set does not match the expected integer 'i',
    // then 'i' is the first missing non-negative integer.
    if (*it != i)
      return i;
  }
  // If the loop completes, it means the set contains a contiguous block of
  // integers from 0 to st.size()-1. The MEX is therefore the next integer.
  return i;
}

/**
 * @brief Solves a single test case.
 */
void solve() {
  int n, tmp;
  cin >> n;
  // Set 'a' will store the unique elements encountered.
  set<int> a;
  // Set 'b' will store elements that are duplicates.
  set<int> b;

  // Block Logic: Read N integers and partition them into sets 'a' and 'b'.
  forn(i, n) {
    cin >> tmp;
    // Check if the element 'tmp' has been seen before.
    if (a.find(tmp) != a.end())
      // If it's already in 'a', it's a duplicate, so add it to 'b'.
      b.insert(tmp);
    else
      // If it's the first time seeing 'tmp', add it to the set of unique elements 'a'.
      a.insert(tmp);
  }

  // Calculate the MEX of the unique elements and the MEX of the duplicates,
  // and print their sum.
  cout << mex(a) + mex(b) << ln;
}

/**
 * @brief Main function that drives the program, handling multiple test cases.
 */
int main() {
  fast_cin();
  int t;
  cin >> t;
  // Loop to process each test case.
  while (t--) {
    solve();
  }
  return 0;
}
