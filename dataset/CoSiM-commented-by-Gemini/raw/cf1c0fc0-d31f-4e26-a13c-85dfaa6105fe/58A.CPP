/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - "Chat room".
 * @author The solution is submitted by a competitive programmer (Mehul Chaturvedi).
 *
 * @section Algorithm
 * The problem is to determine if the string "hello" exists as a subsequence within a
 * given input string. This means the characters 'h', 'e', 'l', 'l', 'o' must appear
 * in the correct order, but not necessarily contiguously.
 *
 * The implemented algorithm uses a brute-force approach with five nested loops. Each
 * loop searches for the next character of "hello" in the substring that follows the
 * previously found character.
 *
 * @section Complexity
 * - Time Complexity: O(L), where L is the length of the input string. Despite the
 *   five nested loops, each character of the input string is visited only a small,
 *   constant number of times. The algorithm is effectively a single linear scan.
 * - Space Complexity: O(L) to store the input string.
 *
 * @note This file contains a large amount of boilerplate code (macros, type aliases,
 * custom hash functions, etc.) typical of competitive programming templates. Most of
 * this code is not used for solving this specific problem.
 */

/*

                Name: Mehul Chaturvedi
                IIT-Guwahati

*/

/*
                Talent is Overrated
*/

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

// --- Boilerplate and Unused Code Section ---
// The following is a large collection of pre-written templates, macros, and
// utility functions for competitive programming. These are not used in this solution.

template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#include <chrono>
#ifndef mehul
#pragma GCC optimize("Ofast")
#endif

typedef long long ll;
typedef long double ld;
typedef unordered_map<int, int> umapii;
typedef unordered_map<int, bool> umapib;
typedef unordered_map<string, int> umapsi;
typedef unordered_map<string, string> umapss;
typedef map<string, int> mapsi;
typedef map<pair<int, int>, int> mappiii;
typedef map<int, int> mapii;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef unordered_set<int> useti;

#define debug(x) cout << '>' << #x << ':' << x << endl;
#define uset unordered_set
#define it iterator
#define mp make_pair
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define f first
#define s second
#define INF 4557430888798830399ll
#define MOD 1000000007
#define EPS 1e-7
#define PI acos(-1)
#define sz(x) (int)(x).size()
template <typename T, typename U> inline bool exist(T &cont, U &val) {
  return cont.find(val) != cont.end();
}
#define fr(i, a, b) for (int i = (a), _b = (b); i <= _b; i++)
#define rep(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define repr(i, n) for (int i = n; i >= 0; i--)
#define frr(i, a, b) for (int i = (a), _b = (b); i >= _b; i--)
#define trav(a, x) for (auto &a : x)
#define fil(ar, val)                                                           
  memset(ar, val,                                                              
         sizeof(ar))

#ifdef mehul
// Debugging templates
template <typename T> void __p(T a) { cout << a << " "; }
template <typename T> void __p(std::vector<T> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename F> void __p(pair<T, F> a) {
  cout << "{ ";
  __p(a.first);
  __p(a.second);
  cout << "}";
}
template <typename T, typename F> void __p(std::vector<pair<T, F>> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename... Arg> void __p(T a1, Arg... a) {
  __p(a1);
  __p(a...);
}
template <typename Arg1> void __f(const char *name, Arg1 &&arg1) {
  cout << name << " : ";
  __p(arg1);
  cout << endl;
}
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args) {
  int bracket = 0, i = 0;
  for (;; i++)
    if (names[i] == ',' && bracket == 0)
      break;
    else if (names[i] == '(')
      bracket++;
    else if (names[i] == ')')
      bracket--;
  const char *comma = names + i;
  cout.write(names, comma - names) << " : ";
  __p(arg1);
  cout << "| ";
  __f(comma + 1, args...);
}
#define trace(...)                                                             
  cout << "Line:" << __LINE__ << " ";                                          
  __f(#__VA_ARGS__, __VA_ARGS__)
int begtime = clock();
#define end_routine()                                                          
  cout << "

Time elapsed: " << fixed                                        
       << double(clock() - begtime) * 1000 / CLOCKS_PER_SEC                    
       << setprecision(12) << " ms

";
#else
#define endl '
'
#define trace(...)
#define end_routine()
#endif


// (Unused) Custom hash for unordered_map
struct custom_hash {
  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }
};

// (Unused) Modular exponentiation
ll poww(ll a, ll b, ll mod) {
  if (b == 0)
    return 1;
  ll ans = poww(a, b / 2, mod);
  if (b % 2 == 0)
    return (ans * ans) % mod;
  return (((ans * ans) % mod) * a) % mod;
}

/**
 * @brief Solves the "Chat room" problem.
 */
void solve() {
  string s;
  cin >> s;

  ll n = s.size();

  // Block Logic: This series of nested loops implements a state machine to find
  // the subsequence "hello". Each level of nesting corresponds to finding the
  // next character in the target word.
  
  // State 1: Find 'h'
  rep(i, n) {
    if (s[i] == 'h') {
      // State 2: Find 'e' after 'h'
      fr(j, i + 1, n - 1) {
        if (s[j] == 'e') {
          // State 3: Find first 'l' after 'e'
          fr(k, j + 1, n - 1) {
            if (s[k] == 'l') {
              // State 4: Find second 'l' after first 'l'
              fr(l, k + 1, n - 1) {
                if (s[l] == 'l') {
                  // State 5: Find 'o' after second 'l'
                  fr(o, l + 1, n - 1) {
                    if (s[o] == 'o') {
                      // Subsequence found, print "YES" and terminate.
                      cout << "YES" << '
';
                      return;
                    }
                  }
                  // If 'o' is not found after the second 'l', the subsequence is impossible.
                  cout << "NO" << '
';
                  return;
                }
              }
              // If second 'l' is not found...
              cout << "NO" << '
';
              return;
            }
          }
          // If first 'l' is not found...
          cout << "NO" << '
';
          return;
        }
      }
      // If 'e' is not found...
      cout << "NO" << '
';
      return;
    }
  }

  // If 'h' is never found in the first place.
  cout << "NO" << '
';
  return;
}

/**
 * @brief Main function, entry point of the program.
 */
int main(int argc, char **argv) {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifdef mehul
  // Redirecting I/O for local testing
  freopen("input.txt", "r", stdin);
#endif

  // The problem has a single test case.
  ll t = 1;
  while (t--) {
    solve();
  }

#ifdef mehul
  end_routine();
#endif

  return 0;
}
