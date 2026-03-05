/**
 * @file 339A.CPP
 * @brief An attempted but non-functional solution to the "Helpful Maths" problem.
 *
 * The goal of the problem is to take a string representing a sum of single-digit
 * numbers (e.g., "3+1+2") and rearrange it so the numbers are in non-decreasing
 * order (e.g., "1+2+3").
 *
 * NOTE: This code is fundamentally broken and does not produce the correct output.
 * The final loop intended to construct the result string never executes because
 * it iterates over an empty vector. The file also contains a large amount of
 * unused template code from competitive programming.
 *
 * @intended_algorithm
 * 1. Read the input string containing the sum (e.g., "3+2+1").
 * 2. Extract all the digits from the string, ignoring the '+' signs.
 * 3. Store these digits as integers in a vector.
 * 4. Sort the vector of integers in non-decreasing order.
 * 5. Reconstruct the output string by joining the sorted numbers with '+' signs.
 * 6. Print the reconstructed string.
 *
 * @complexity (of a correct algorithm)
 * Time Complexity: O(L log L), where L is the length of the string, dominated
 * by the sorting of the numbers. Since the numbers are single digits, counting
 * sort could achieve O(L).
 * Space Complexity: O(L) to store the numbers.
 */

// --- Boilerplate and Unused Code Section ---
// The following includes, macros, and typedefs are common in competitive
// programming templates but are not used in the logic of this specific file.
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;

#define gc getchar_unlocked
#define ff first
#define ss second
#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int, int>
#define vi vector<int>
#define mii map<int, int>
#define pqb priority_queue<int>
#define pqs priority_queue<int, vi, greater<int>>
#define setbits(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define ps(x, y) fixed << setprecision(y) << x
#define mk(arr, n, type) type *arr = new type[n];
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define all(x) x.begin(), x.end()
#define clr(x) memset(x, 0, sizeof(x))
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define toDigit(c) (c - '0')

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    pbds;

void loadInputs() {
  // fast input-output
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
}
// --- End of Boilerplate Section ---

int32_t main() {
  loadInputs();
  string s;
  cin >> s;

  // This vector is an unnecessary intermediate copy of the string.
  vector<char> arr(s.begin(), s.end());
  
  // `arr1` will correctly store the integer digits from the input.
  vi arr1;
  
  // This loop correctly extracts the digits from the input string.
  for (size_t i = 0; i < arr.size(); i++) {
    if (arr[i] == '+') {
      continue;
    } else {
      // Convert char digit to int and push to vector.
      arr1.pb(arr[i] - '0');
    }
  }

  // This correctly sorts the digits in non-decreasing order.
  sort(arr1.begin(), arr1.end());
  
  // `arr2` is intended to hold the characters for the final output string.
  vector<char> arr2;

  // !!! BUG !!!
  // The loop condition `i < arr2.size()` is immediately false because `arr2`
  // was just created and is empty. The body of this loop will NEVER execute.
  for (size_t i = 0; i < arr2.size(); i++) {
    // This code is also logically incorrect. It should be iterating over the
    // sorted numbers in `arr1` and building the string with '+' signs.
    arr2.pb(static_cast<char>(arr[i]));
    arr2.pb('+');
  }

  // Because the loop above never runs, `arr2` remains empty.
  // `ans` is therefore constructed as an empty string.
  string ans(arr2.begin(), arr2.end());

  // The program prints an empty string.
  cout << ans << "\n";

  return 0;
}
