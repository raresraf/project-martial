/***********************************************************|
|          /\           /-------|   -----    |-----------|  |
|         /  \         /              |      |              |
|        /    \       |               |      |              |
|       /      \      |----|          |      |-------|      |
|      / ------ \          |-----|    |      |              |
|     /          \               |    |      |              |
|    /            \              /    |      |              |
|  ---            --- |---------/   -----  -----            |
|                                                           |
|   Codeforces = Asif_Watson  ||  Codechef = asif_watson    |
|   Mail = ashfaqislam33@gmail.com                          |
|   CSTE, Noakhali Science and Technology University(11)    |
|                                                           |
************************************************************/

#include "bits/stdc++.h"
using namespace std;

#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define all(v) v.begin(), v.end()
#define allre(v) v.rbegin(), v.rend()
#define sp(x, y) fixed << setprecision(y) << x
#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))

const double pi = acos(-1.0);
const double EPS = 1e-6;
const int MOD = (int)1e9 + 7;

long long bigmod(long long b, long long p, long long m) {
  if (p == 0)
    return 1;
  if (p % 2 == 0) {
    long long c = bigmod(b, p / 2, m);
    return ((c % m) * (c % m)) % m;
  } else
    return ((b % m) * (bigmod(b, p - 1, m))) % m;
}

template <class T> inline T gcd(T x, T y) {
  if (y == 0)
    return x;
  return gcd(y, x % y);
}
template <class T> inline T lcm(T x, T y) { return ((x / gcd(x, y)) * y); }

bool Reverse(long long a, long long b) { return a > b; }

bool compare(pair<long long, long long> a, pair<long long, long long> b) {
  if (a.first == b.first)
    return a.second > b.second;
  return a.first < b.first;
}

int main() {
  IOS

      int n;
  cin >> n;

  int Array[n], Sum = 0;

  for (int i = 0; i < n; i++) {
    cin >> Array[i];
    Sum += Array[i];
  }

  sort(Array, Array + n, Reverse);

  int Ans = 0, Minus = 0;
  for (int i = 0; i < n; i++) {
    if (Minus > Sum - Minus)
      break;

    Minus += Array[i];
    Ans = i + 1;
  }

  cout << Ans << endl;

  return 0;
}
