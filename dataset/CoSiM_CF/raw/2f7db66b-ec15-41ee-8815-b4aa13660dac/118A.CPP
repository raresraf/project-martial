// Link to Problem: https://codeforces.com/problemset/problem/118/A

#include <bits/stdc++.h>

#define ll long long
#define ull unsigned long long int
#define ld long double
#define mod 1000000007
#define setbits(x) __builtin_popcount(x)
#define setbits_ll(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctz(x)
#define zrobits_ll(x) __builtin_ctzll(x)
#define inf 1e18
#define ps(x, y)                                                               \
  fixed << setprecision(y)                                                     \
        << x //set precision of double x to y decimal places \
                                               //ps(1.254754, 4) -> 1.2547

using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, ans = "";
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    char x = s[i];
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'A' ||
        x == 'E' || x == 'I' || x == 'O' || x == 'U' || x == 'y' || x == 'Y')
      continue;

    ans += ".";
    ans += tolower(x);
  }

  cout << ans << endl;
}
// functions

int getpow(int a, int b) {
  if (b == 0)
    return 1;

  int x = getpow(a, b / 2);
  if (b & 1)
    return x * x * a;
  else
    return x * x;
}