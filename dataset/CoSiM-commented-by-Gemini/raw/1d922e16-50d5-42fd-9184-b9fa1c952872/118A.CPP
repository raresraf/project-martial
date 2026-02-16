/**
 * @file 118A.CPP
 * @brief Solution for Codeforces problem 118A, "String Task".
 *
 * @description
 * This program processes a string according to a specific set of rules:
 * 1. Convert all characters to lowercase.
 * 2. Delete all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 3. Before each remaining character (consonant), insert a '.'.
 *
 * The file also contains a large template of common competitive programming
 * helper functions, which are not used for this particular problem.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Convert the entire string to lowercase to simplify vowel checking.
 * 3. Initialize an empty string to build the result.
 * 4. Iterate through each character of the lowercase string.
 *    a. If the character is a vowel (including 'y'), it is ignored.
 *    b. If the character is a consonant, a '.' is appended to the result
 *       string, followed by the consonant itself.
 * 5. Print the final result string.
 *
 * @time_complexity O(L), where L is the length of the input string.
 * @space_complexity O(L) to store the original and result strings.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#define mod 1000000007
#define Debug(x) cout << '>' << #x << ':' << x << endl;
#define FOR(i, a, b) for (long long int i = (a); i < (b); i++)
#define REP(i, a) for (long long int i = (0); i < (a); i++)
const int INF = 1 << 29;
typedef long long int ll;
typedef unsigned long long int ull;

/* Start of unused competitive programming template functions */
ll gcd(ll a, ll b) { return (a % b == 0) ? abs(b) : gcd(b, a % b); }
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }
ll max(ll a, ll b) {
  if (a > b)
    return a;
  return b;
}
ll power(ll x, ll y) {
  ll res = 1;
  while (y > 0) {
    if (y & 1)
      res = res * x;
    y = y >> 1;
    x = x * x;
  }
  return res;
}
ll powermod(ll x, ll y) {
  ll res = 1;
  x = x % mod;
  while (y > 0) {
    if (y & 1)
      res = (res * x) % mod;
    y = y >> 1;
    x = (x * x) % mod;
  }
  return res;
}
bool isPrime(ll n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (int i = 5; i * i <= n; i += 6) {
    if ((n % i == 0) || (n % (i + 2) == 0))
      return false;
  }
  return true;
}
ll nCr(ll n, ll r) {
  if (r == 0)
    return 1;
  return (n * nCr(n - 1, r - 1)) / r;
}
ll countDivisors(ll n) {
  ll cnt = 0;
  for (int i = 1; i <= sqrt(n); i++) {
    if (n % i == 0) {
      if (n / i == i)
        cnt++;
      else
        cnt = cnt + 2;
    }
  }
  return cnt;
}
ll modulo(ll a, ll b) {
  ll r = a % b;
  return r < 0 ? r + b : r;
}
ll bsearch_array(ll n, ll *arr, ll first, ll last) {
  if (first == last) {
    if (arr[first] == n)
      return first + 1;
    else
      return -1;
  } else {
    ll m = (first + last) / 2;
    if (arr[m] == n)
      return m + 1;
    else {
      if (arr[m] < n)
        return bsearch_array(n, arr, m + 1, last);
      else
        return bsearch_array(n, arr, first, m);
    }
  }
}
ll bsearch_vect(std::vector<ll> &nums, int target) {

  if (nums.size() == 0)
    return -1;

  int left = 0, right = nums.size() - 1;
  while (left < right) {
    ll mid = left + (right - left) / 2;
    if (nums[mid] == target) {
      return mid;
    } else if (nums[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
/* End of unused competitive programming template functions */
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from C standard streams for faster I/O.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s;
  cin >> s;
  // Convert the entire string to lowercase to simplify vowel checking.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  string ans = "";
  // Block Logic: Iterate through the string to build the result.
  FOR(i, 0, s.length()) {
    // Pre-condition: Check if the character is a vowel. In this problem, 'y' is a vowel.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue; // If it's a vowel, skip it.

    // If it's a consonant, prepend a '.' and add it to the result.
    ans.append(".");
    ans.append(1, s[i]);
  }
  cout << ans << endl;

  return 0;
}
