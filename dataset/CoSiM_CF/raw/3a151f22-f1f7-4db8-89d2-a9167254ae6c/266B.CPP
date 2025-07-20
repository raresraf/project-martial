#include <bits/stdc++.h>
typedef long long ll;
typedef double ld;
#define vll vector<ll>
#define vvll vector<vll>
#define vld vector<ld>
#define vvld vector<vld>
#define pll pair<ll, ll>
#define vllp vector<pll>
#define MOD 1000000007
#define endl "\n"
#define MAX 1000000007
#define nl << endl
#define f(i, a, b) for (i = a; i < b; i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  int b = 0;
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == 'B') {
      b = 1;
    }
  }
  if (!b) {
    cout << s;
    return 0;
  }

  while (t--) {
    int a[n] = {0};
    for (int i = 0; i < n; ++i) {
      if (s[i] == 'B') {
        a[i] = 1;
      }
    }
    for (int i = 0; i < n - 1; ++i) {
      if (s[i] == 'B' && s[i + 1] == 'G' && a[i] == 1) {
        s[i + 1] = 'B';
        s[i] = 'G';
        a[i] = 0;
      }
    }
  }
  cout << s;
}
