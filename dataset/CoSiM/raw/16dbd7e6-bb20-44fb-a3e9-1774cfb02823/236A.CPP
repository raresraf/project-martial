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
const ll mod = 1e9 + 7;

int main() {
  std::ios_base::sync_with_stdio(0);
  cin.tie(0);
  string s;
  cin >> s;
  vi cnt(26, 0);
  for (int i = 0; i < s.size(); ++i)
    cnt[s[i] - 'a']++;
  int c = 0;
  for (int i = 0; i < 26; ++i) {
    c += (cnt[i] != 0);
  }
  if (c % 2)
    cout << "IGNORE HIM!\n";
  else
    cout << "CHAT WITH HER!\n";
  return 0;
}