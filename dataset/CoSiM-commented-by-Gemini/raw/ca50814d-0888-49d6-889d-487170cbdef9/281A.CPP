/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - "Word Capitalization".
 * @author The solution is submitted by a competitive programmer (_MONU KUMAR).
 *
 * @section Algorithm
 * The problem requires capitalizing the first letter of a given word.
 * The algorithm is straightforward:
 * 1. Read the input word as a string.
 * 2. Check if the first character of the string is a lowercase letter.
 * 3. If it is, convert it to its uppercase equivalent.
 * 4. Print the modified string.
 *
 * @section Complexity
 * - Time Complexity: O(L) for reading and printing the string of length L. The capitalization itself is O(1).
 * - Space Complexity: O(L) to store the input string.
 *
 * @note A significant portion of this file consists of unused, generic boilerplate code
 * and macros common in competitive programming for solving a wide range of problems. The
 * core logic for this specific problem is confined to a few lines in the main function.
 */
/////*31022618*/////
//_MONU KUMAR
#include <bits/stdc++.h>
using namespace std;

// --- Competitive Programming Boilerplate ---
// This section contains a large number of pre-written macros, type aliases,
// and utility functions to speed up coding during programming contests.
// Most of these are not used in the solution for this particular problem.

#define fast                                                                   
  ios_base::sync_with_stdio(false);                                            
  cin.tie(0);                                                                  
  cout.tie(0);
#define tc                                                                     
  ll t = 1;                                                                    
  cin >> t;                                                                    
  while (t--)
#define ll long long int
#define ld long double
#define ull unsigned long long
#define pll pair<ll, ll>
#define vi std::vector<ll>
#define vp vector<pll>
#define mll std::map<ll, ll>
#define mpl                                                                    
  std:                                                                         
  map<pll, ll>
#define p1(x) cout << x << "
";
#define p2(x, y) cout << x << " " << y << "
";
#define p3(x, y, z) cout << x << " " << y << " " << z << "
";
#define p4(x, y, z, w) cout << x << " " << y << " " << z << " " << w << "
";
#define trace(x)                                                               
  cout << #x << ": " << x << " "                                               
       << "
";
#define trace2(x, y)                                                           
  cout << #x << ": " << x << " | " << #y << ": " << y << "
";
#define trace3(x, y, z)                                                        
  cout << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    
       << z << "
";
#define trace4(a, b, c, d)                                                     
  cout << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": "   
       << c << " | " << #d << ": " << d << "
";
#define nL cout << "
";
#define lp(i, l, r) for (long long int i = l; i < r; i++)
#define lpr(i, r, l) for (long long int i = r - 1; i >= l; i--)
#define mem(a, x) memset(a, x, sizeof(a))
#define pi 3.1415926535897932384626
#define mod 1000000007
#define big 1e18
#define small -big
#define mp(a, b) make_pair(a, b)
#define pb(x) push_back(x)
#define ff first
#define ss second
#define siz 256
#define leadzero(a) __builtin_clz(a)
#define trailzero(a) __builtin_ctz(a)
#define bitcount(a) __builtin_popcount(a)
#define all(v) v.begin(), v.end()
#define lps(v, x) (lower_bound(all(v), x) - v.begin())
#define ups(v, x) (upper_bound(all(v), x) - v.begin())
#define ina(a, n)                                                              
  ll a[n];                                                                     
  lp(i, 0, n) cin >> a[i];
#define ona(a, n) lp(i, 0, n) cout << a[i] << " ";

// --- Unused Utility Functions ---

/**
 * @brief (Unused) Checks if two C-style strings are anagrams.
 */
bool anagram(char *s1, char *s2) {
  ll count[siz];
  mem(count, 0);
  ll i;
  for (i = 0; s1[i] && s2[i]; i++) {
    count[s1[i]]++;
    count[s2[i]]--;
  }
  if (s1[i] || s2[i]) {
    return false;
  }
  lp(i, 0, siz) {
    if (count[i]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief (Unused) Calculates (a * b) % zz for large numbers.
 */
ll moduloMultiplication(ll a, ll b, ll zz) {
  ll res = 0;
  a %= mod;
  while (b) {
    if (b & 1)
      res = (res + a) % zz;
    a = (2 * a) % zz;
    b >>= 1;
  }
  return res;
}

/**
 * @brief (Unused) Converts a numeric string to a long long.
 */
ll convert(string s) {
  bool o = true;
  ll ans = 0;
  for (ll i = 0; i < s.length(); i++) {
    ans = ans * 10 + (s[i] - 48);
  }
  return ans;
}

/**
 * @brief (Unused) Calculates (a^b) % modulo using binary exponentiation.
 */
ll powmod(ll a, ll b, ll modulo) {
  if (b == 0 || a == 1)
    return 1;
  ll half = powmod(a, (b / 2), modulo) % modulo;
  ll full = (half * half) % modulo;

  if (b % 2)
    return (full * a) % modulo;
  return full % modulo;
}

/**
 * @brief (Unused) Calculates modular multiplicative inverse using Fermat's Little Theorem.
 */
ll invmod(ll a, ll modulo) {
  ll check = powmod(a, modulo - 2, modulo) % modulo;
  return check;
}

/**
 * @brief (Unused) Collection of self-defined min/max and math functions.
 */
ll max(ll a, ll b) {
  if (a > b) {
    return a;
  }
  return b;
}
ll min(ll a, ll b) {
  if (a < b) {
    return a;
  }
  return b;
}
ll min(ll a, ll b, ll c) {
  if (a < b) {
    return min(a, c);
  } else {
    return min(b, c);
  }
}
ll max(ll a, ll b, ll c) {
  if (a < b) {
    return max(b, c);
  } else {
    return min(a, c); // Bug: Should be max(a, c)
  }
}
ll min(ll a, int b) {
  if (a < b)
    return a;
  return b;
}
ll min(int a, ll b) {
  if (a < b)
    return a;
  return b;
}
ll max(ll a, int b) {
  if (a > b)
    return a;
  return b;
}
ll max(int a, ll b) {
  if (a > b)
    return a;
  return b;
}
ll gcd(ll a, ll b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
void yes() { cout << "YES" << "
"; }
void no() { cout << "NO" << "
"; }

//###########################################################################################
int main() {
  // Functional Utility: A macro to disable synchronization with C stdio and untie cin/cout,
  // speeding up I/O operations in competitive programming.
  fast

  // A large number of unused variable declarations, common in template-based coding.
  ll n,
      k, ans = 0, flag = 0, temp = 0, f1 = 0, count = 0, f2 = 0, sum = 0,
         xx = 0, minn1 = big, minn2 = big, maxx1 = small, maxx2 = small;
  
  string s;
  cin >> s;

  /**
   * @brief Core logic for "Word Capitalization".
   * This block checks if the first character is lowercase ('a' through 'z') and,
   * if so, converts it to uppercase.
   */
  if (s[0] >= 'a' && s[0] <= 'z') {
    // ASCII-based conversion: The uppercase equivalent of a lowercase letter
    // is found by subtracting 32 from its ASCII value.
    s[0] = (char)((int)s[0] - 32);
  }

  // Print the resulting string.
  p1(s)

  return 0;
}
//////////*********************end of program*********************//////////
