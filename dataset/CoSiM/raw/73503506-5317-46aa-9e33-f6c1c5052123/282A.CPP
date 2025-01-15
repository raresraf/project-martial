// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~ Aashutosh Rathi~~~~~~
// ~~~~~~~~aashutoshrathi~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Use memcpy while copying matrix :D
// o_O Never do calculation in if block -_-
// Use __builtin_popcount(z) to get set bit count of z

#include <bits/stdc++.h> // Apne aap me sab kuch hai isme
#include <stdio.h>

using namespace std; // Kyuki yahi mai-baap hai

#define fj(a) for (ll j = 0; j < a; ++j)
#define fi(a) for (ll i = 0; i < a; ++i)
#define fiz(a) for (ll i = 1; i <= a; ++i)
#define mod(x) x % 1000000007
#define MIN(n1, n2) ((n1) > (n2) ? (n2) : (n1))
#define MAX(n1, n2) ((n1) > (n2) ? (n1) : (n2))
#define MID(s, e) (s + (e - s) / 2)
#define tezzzzz                                                                \
  fin;                                                                         \
  fout;                                                                        \
  fio;
#define pb push_back
#define fin cin.tie(NULL)
#define fout cout.tie(NULL)
#define fio ios::sync_with_stdio(false)
typedef pair<long, long> lpairs;
typedef long long ll;
#define a first
#define b second

int main() {
  tezzzzz ll n;
  cin >> n;
  int x = 0;
  string s;
  fi(n) {
    cin >> s;
    if (s[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;
  return 0;
}