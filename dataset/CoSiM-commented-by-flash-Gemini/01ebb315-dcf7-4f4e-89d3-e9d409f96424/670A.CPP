// please refer from line no. 207
/////*31022618*/////
//_MONU KUMAR
/**
 * @file 670A.CPP
 * @brief Competitive programming solution template and problem-specific logic.
 * This file contains a common competitive programming template with various
 * utility functions and macros, followed by the solution for a specific problem
 * within the `main` function. The problem appears to involve counting occurrences
 * based on a modulo 7 condition over a given range `n`.
 *
 * Algorithm: Direct counting based on modulo arithmetic.
 * Time Complexity: O(N) where N is the input integer.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define tc                                                                     \
  ll t = 1;                                                                    \
  cin >> t;                                                                    \
  while (t--)
#define ll long long int
#define ld long double
#define ull unsigned long long
#define pll pair<ll, ll>
#define vi std::vector<ll>
#define vp vector<pll>
#define mll std::map<ll, ll>
#define mpl                                                                    \
  std:                                                                         \
  map<pll, ll>
#define p1(x) cout << x << "\n";
#define p2(x, y) cout << x << " " << y << "\n";
#define p3(x, y, z) cout << x << " " << y << " " << z << "\n";
#define p4(x, y, z, w) cout << x << " " << y << " " << z << " " << w << "\n";
#define trace(x)                                                               \
  cout << #x << ": " << x << " "                                               \
       << "\n";
#define trace2(x, y)                                                           \
  cout << #x << ": " << x << " | " << #y << ": " << y << "\n";
#define trace3(x, y, z)                                                        \
  cout << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << "\n";
#define trace4(a, b, c, d)                                                     \
  cout << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": "   \
       << c << " | " << #d << ": " << d << "\n";
#define nL cout << "\n";
#define lp(i, l, r) for (long long int i = l; i < r; i++)
#define lpr(i, r, l) for (long long int i = r - 1; i >= l; i--)
#define mem(a, x) memset(a, x, sizeof(a)) // set elements of array to some value
#define pi 3.1415926535897932384626
#define mod 1000000007
#define big 1e18
#define small -big
#define mp(a, b) make_pair(a, b)
#define pb(x) push_back(x)
#define ff first
#define ss second
#define siz 256
#define leadzero(a) __builtin_clz(a) // count leading zeroes
#define trailzero(a) __builtin_ctz(a) // count trailing zeroes
#define bitcount(a) __builtin_popcount(a) // count set bits
#define all(v) v.begin(), v.end()
#define lps(v, x) (lower_bound(all(v), x) - v.begin())
#define ups(v, x) (upper_bound(all(v), x) - v.begin())
#define ina(a, n)                                                              \
  ll a[n];                                                                     \
  lp(i, 0, n) cin >> a[i];
#define ona(a, n) lp(i, 0, n) cout << a[i] << " ";
//#########################################################################################
bool anagram(char *s1, char *s2) {
  /// Functional Utility: Checks if two C-style strings are anagrams of each other.
  /// This function uses a character count array to determine if both strings
  /// contain the same characters with the same frequencies.
  ll count[siz];
  mem(count, 0);
  ll i;
  // Block Logic: Populate character counts for both strings.
  // Precondition: s1 and s2 are null-terminated character arrays.
  // Invariant: After each iteration, `count` reflects the difference in character frequencies
  // for the processed prefix of s1 and s2.
  for (i = 0; s1[i] && s2[i]; i++) {
    count[s1[i]]++;
    count[s2[i]]--;
  }
  // Conditional Logic: If strings have different lengths (one is not yet null-terminated while the other is), they cannot be anagrams.
  if (s1[i] || s2[i]) {
    return false;
  }
  // Block Logic: Verify all character counts are zero, indicating equal frequencies for all characters.
  // Precondition: Both strings have been fully traversed and found to be of equal length.
  // Invariant: If any character's count is non-zero, the strings are not anagrams.
  lp(i, 0, siz) {
    if (count[i]) {
      return false;
    }
  }
  return true;
}
//###########################################################################################
// modulo Multiplication
ll moduloMultiplication(ll a, ll b, ll zz) {
  /// Functional Utility: Computes (a * b) % zz using binary multiplication to prevent overflow.
  /// This function efficiently calculates the product modulo `zz` without intermediate large values.
  ll res = 0;
  a %= mod;
  // Block Logic: Perform binary multiplication.
  // Precondition: `a`, `b`, and `zz` are non-negative long long integers. `zz` is the modulus.
  // Invariant: `res` accumulates (a * current_bit_of_b) % zz.
  while (b) {
    // Conditional Logic: If the current bit of `b` is set, add `a` to `res`.
    if (b & 1)
      res = (res + a) % zz;

    a = (2 * a) % zz;

    b >>= 1;
  }
  return res;
}
//###########################################################################################

ll convert(string s) {
  /// Functional Utility: Converts a string representation of a number to a long long integer.
  /// This function iterates through the string, accumulating the numeric value.
  bool o = true; // Variable 'o' appears unused, consider removing for clarity.
  ll ans = 0;
  // Block Logic: Convert each character digit to its numeric value and accumulate.
  // Precondition: `s` contains only digit characters ('0'-'9').
  // Invariant: `ans` holds the numeric value of the processed prefix of the string.
  for (ll i = 0; i < s.length(); i++) {
    ans = ans * 10 + (s[i] - 48); // Inline: Convert character to integer using ASCII offset.
  }
  return ans;
}
//###########################################################################################
// function to calculate a^b under mod...
ll powmod(ll a, ll b, ll modulo) {
  /// Functional Utility: Computes (a^b) % modulo using binary exponentiation (exponentiation by squaring).
  /// This function efficiently calculates large powers modulo a given number.
  // Conditional Logic: Base cases for exponentiation: a^0 = 1 or 1^b = 1.
  if (b == 0 || a == 1)
    return 1;
  // Block Logic: Recursive step for binary exponentiation.
  // Precondition: `a`, `b`, and `modulo` are non-negative long long integers. `modulo` is the modulus.
  // Invariant: `half` stores (a^(b/2)) % modulo.
  ll half = powmod(a, (b / 2), modulo) % modulo;
  ll full = (half * half) % modulo;

  // Conditional Logic: Handle odd exponents by multiplying with `a` one more time.
  if (b % 2)
    return (full * a) % modulo;
  return full % modulo;
}
//###########################################################################################
// function to calculate inverse modulo under mod...assuming gcd is 1
ll invmod(ll a, ll modulo) {
  /// Functional Utility: Computes the modular multiplicative inverse of 'a' modulo 'modulo'.
  /// This function uses Fermat's Little Theorem (a^(modulo-2) % modulo) for prime moduli.
  /// Precondition: `modulo` must be a prime number and `gcd(a, modulo) == 1`.
  ll check = powmod(a, modulo - 2, modulo) % modulo;
  return check;
}
//###########################################################################################

ll max(ll a, ll b) {
  /// Functional Utility: Returns the maximum of two long long integers.
  // Conditional Logic: Compare `a` and `b` to determine the maximum.
  if (a > b) {
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b) {
  /// Functional Utility: Returns the minimum of two long long integers.
  // Conditional Logic: Compare `a` and `b` to determine the minimum.
  if (a < b) {
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b, ll c) {
  /// Functional Utility: Returns the minimum of three long long integers.
  // Conditional Logic: Compare `a` and `b`, then compare the result with `c` to find the overall minimum.
  if (a < b) {
    return min(a, c);
  } else {
    return min(b, c);
  }
}
//###########################################################################################
ll max(ll a, ll b, ll c) {
  /// Functional Utility: Returns the maximum of three long long integers.
  // Conditional Logic: Compare `a` and `b`, then compare the result with `c` to find the overall maximum.
  // Note: The original code has a logical error here, returning min(a,c) if a > b. This is being commented as-is due to zero-code mutation.
  if (a < b) {
    return max(b, c);
  } else {
    return min(a, c); // Bug: Should be max(a, c) or (a > c ? a : c)
  }
}
//###########################################################################################
ll min(ll a, int b) {
  /// Functional Utility: Returns the minimum of a long long and an integer.
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll min(int a, ll b) {
  /// Functional Utility: Returns the minimum of an integer and a long long.
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll max(ll a, int b) {
  /// Functional Utility: Returns the maximum of a long long and an integer.
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll max(int a, ll b) {
  /// Functional Utility: Returns the maximum of an integer and a long long.
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll gcd(ll a, ll b) {
  /// Functional Utility: Computes the Greatest Common Divisor (GCD) of two long long integers using the Euclidean algorithm.
  // Precondition: `a` and `b` are non-negative long long integers.
  // Conditional Logic: Base case for recursion: GCD(a, 0) = a.
  if (b == 0)
    return a;
  // Block Logic: Recursive step of Euclidean algorithm: GCD(a, b) = GCD(b, a % b).
  return gcd(b, a % b);
}
//###########################################################################################
ll lcm(ll a, ll b) {
  /// Functional Utility: Computes the Least Common Multiple (LCM) of two long long integers.
  /// Precondition: `a` and `b` are non-negative long long integers.
  /// Postcondition: Returns `(a * b) / gcd(a, b)`.
  // Block Logic: Calculate LCM using the formula: LCM(a, b) = |a * b| / GCD(a, b).
  // This implementation divides by GCD first to prevent potential overflow of `a * b`.
  return a / gcd(a, b) * b;
}
//###########################################################################################
void yes() {
  /// Functional Utility: Prints "YES" followed by a newline to standard output.
  cout << "YES"
       << "\n";
}
//###########################################################################################
void no() {
  /// Functional Utility: Prints "NO" followed by a newline to standard output.
  cout << "NO"
       << "\n";
}
//###########################################################################################
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  fast
      //  clock_t launch=clock();
      //  string r = s1.substr(1, 3);
      //  while(clock()<CLOCKS_PER_SEC*10)
      //  using num = modnum<int(1e9)+7>;
      //  num len = s.size();
      //  double PI= 2 * acos(0.0); or M_PI
      int n;
  int x = 0, y = 0;
  // Block Logic: Read the input integer `n`.
  scanf("%d", &n);
  // Block Logic: Iterate from 0 to n-1 to simulate days and count specific weekend days.
  // Precondition: `n` is a non-negative integer representing the total number of days.
  // Invariant: `x` counts the occurrences where `i % 7` is 5 or 6 (e.g., Saturday, Sunday).
  // Invariant: `y` counts the occurrences where `i % 7` is 0 or 1 (e.g., Monday, Tuesday).
  // The problem implies a mapping of days to `i % 7` where 0 and 1 represent the first two days of a week,
  // and 5 and 6 represent the last two days (e.g., Saturday and Sunday).
  for (int i = 0; i < n; i++) {
    // Conditional Logic: Check if the current day `i` falls on the 6th (index 5) or 7th (index 6) day of a 7-day cycle.
    if (i % 7 == 5 || i % 7 == 6)
      x++;
    // Conditional Logic: Check if the current day `i` falls on the 1st (index 0) or 2nd (index 1) day of a 7-day cycle.
    if (i % 7 == 0 || i % 7 == 1)
      y++;
  }
  // Block Logic: Print the final counts of `x` and `y` separated by a space, followed by a newline.
  printf("%d %d\n", x, y);

  // clog<<((long double)(clock()-launch)/CLOCKS_PER_SEC)<<"\n";
  return 0;
}
//////////*********************end of program*********************//////////
