#include <bits/stdc++.h>
#define fs first
#define sc second
#define pi 3.141592653589793238462643383279502884197
//#define e 2.7182818284590452353602874713526624977572

typedef long long int ll;
typedef long double ld;
typedef std::vector<ll> vi;
typedef std::vector<bool> vb;
typedef std::vector<char> vc;
typedef std::pair<ll, ll> pii;
typedef std::vector<pii> vii;

using namespace std;
const int mod = 1e9 + 7;

int main() {
  std::ios::sync_with_stdio(0);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.size() > 10)
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << '\n';
    else
      cout << s << '\n';
  }
}