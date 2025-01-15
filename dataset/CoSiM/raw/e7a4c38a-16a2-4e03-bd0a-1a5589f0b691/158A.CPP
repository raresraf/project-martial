// .... .... .... !

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#define rep(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define fer(i, x, n) for (int i = (x), _n = (n); i < _n; i++)
#define rof(i, n, x) for (int i = (n), _x = (x); i-- > _x;)
#define fch(i, x) for (__typeof(x.begin()) i = x.begin(); i != x.end(); i++)
#define all(x) x.begin(), x.end()
#define sz(x) int((x).size())
#define cmx(a, b) (a = max(a, b))
#define cmn(a, b) (a = min(a, b))
#define pb push_back
#define mp make_pair
#define X first
#define Y second

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

////////////////////////////////////////////////

int n, k;
int a[55];
int ans = 0;

int main() {
  cin >> n >> k;
  k--;
  rep(i, n) cin >> a[i];
  rep(i, n) if (a[i] <= 0 || a[i] < a[k]) break;
  else ans++;
  cout << ans << endl;
  {
    int _;
    cin >> _;
  }
}
