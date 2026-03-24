/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - "Way Too Long Words".
 * @details This file contains a typical competitive programming setup to solve the problem.
 * The problem requires shortening words longer than 10 characters into an abbreviation:
 * the first character, the number of middle characters, and the last character.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. The number of test cases `t` is read.
 * 2. A loop runs `t` times.
 * 3. Inside the loop, a string `x` is read.
 * 4. The length of `x` is checked.
 * 5. If the length is greater than 10, the abbreviation is printed.
 * 6. Otherwise, the original string is printed.
 *
 * @note This submission is characteristic of competitive programming, featuring a large
 * number of pre-defined macros, type definitions, and a fast I/O setup to
 * accelerate development and execution speed.
 *
 * @complexity
 * Time Complexity: O(T * L_avg), where T is the number of test cases and L_avg is the average
 * length of the words. Since processing each word is very fast, it's effectively O(T).
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <bits/stdc++.h> // Include every standard library
using namespace std;

// Type definitions for common data structures to shorten code.
typedef long long LL;
typedef pair<int, int> pii;
typedef pair<LL, LL> pll;
typedef pair<string, string> pss;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
typedef vector<LL> vl;
typedef vector<vl> vvl;

// Constants used in competitive programming contexts.
double EPS = 1e-9;
int INF = 1000000005;
long long INFF = 1000000000000000005LL;
double PI = acos(-1);
// Direction arrays for grid-based problems.
int dirx[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
int diry[8] = {0, 1, -1, 0, -1, 1, -1, 1};

// A large collection of macros for debugging, looping, and common operations.
#define DEBUG fprintf(stderr, "====TESTING====\n")
#define VALUE(x) cerr << "The value of " << #x << " is " << x << endl
#define debug(...) fprintf(stderr, __VA_ARGS__)
#define FOR(a, b, c) for (int(a) = (b); (a) < (c); ++(a))
#define FORN(a, b, c) for (int(a) = (b); (a) <= (c); ++(a))
#define FORD(a, b, c) for (int(a) = (b); (a) >= (c); --(a))
#define FORSQ(a, b, c) for (int(a) = (b); (a) * (a) <= (c); ++(a))
#define FORC(a, b, c) for (char(a) = (b); (a) <= (c); ++(a))
#define FOREACH(a, b) for (auto &(a) : (b))
#define REP(i, n) FOR(i, 0, n)
#define REPN(i, n) FORN(i, 1, n)
#define MAX(a, b) a = max(a, b)
#define MIN(a, b) a = min(a, b)
#define SQR(x) ((LL)(x) * (x))
#define RESET(a, b) memset(a, b, sizeof(a))
#define fi first
#define se second
#define mp make_pair
#define pb push_back
#define ALL(v) v.begin(), v.end()
#define ALLA(arr, sz) arr, arr + sz
#define SIZE(v) (int)v.size()
#define SORT(v) sort(ALL(v))
#define REVERSE(v) reverse(ALL(v))
#define SORTA(arr, sz) sort(ALLA(arr, sz))
#define REVERSEA(arr, sz) reverse(ALLA(arr, sz))
#define PERMUTE next_permutation
#define TC(t) while (t--) // Macro for handling test cases.
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL) // Macro for fast I/O optimization.

int main() {
  FAST_INP;
  int t;
  cin >> t;
  // The TC(t) macro expands to `while(t--)`, processing `t` test cases.
  TC(t) {

    string x;
    cin >> x;
    /**
     * Block Logic: If the word's length is greater than 10, it's abbreviated.
     * Otherwise, it's printed as is.
     */
    if (x.length() > 10)
      // The abbreviation is the first letter, the count of middle letters, and the last letter.
      cout << x[0] << x.length() - 2 << x[x.length() - 1] << "\n";
    else
      cout << x << "\n";
  }
}