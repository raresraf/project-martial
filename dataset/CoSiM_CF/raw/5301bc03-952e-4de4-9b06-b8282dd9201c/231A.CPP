// .... .... .... !

#include <fstream>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#include <cstdio>
#include <cstring>

using namespace std;

#define rep(i, n) for (int i = 0, _n = (int)(n); i < _n; i++)
#define fer(i, x, n) for (int i = (int)(x), _n = (int)(n); i < _n; i++)
#define rof(i, n, x) for (int i = (int)(n), _x = (int)(x); i-- > _x;)
#define fch(i, x) for (__typeof(x.begin()) i = x.begin(); i != x.end(); i++)
#define sz(x) (int((x).size()))
#define X first
#define Y second

typedef long long ll;
typedef pair<int, int> pii;

////////////////////////////////////////////////////////////////////////////////

int main() {
  ios::sync_with_stdio(false);
  int n;
  cin >> n;
  int r = 0;
  rep(i, n) {
    int s, x, y, z;
    cin >> x >> y >> z;
    s = x + y + z;
    if (s >= 2)
      r++;
  }
  cout << r << endl;
  {
    int _;
    cin >> _;
  }
  return 0;
}
