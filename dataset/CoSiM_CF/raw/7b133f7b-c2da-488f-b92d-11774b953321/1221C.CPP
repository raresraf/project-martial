#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>
#define N 200005
//#define M 10005
#define inf 1000000000
#define pii pair<int, int>
using namespace std;
typedef long long ll;
const ll mod = 1000000007LL;
int q;
ll c, m, x;
int main() {
  cin >> q;
  for (int cas = 1; cas <= q; ++cas) {
    cin >> c >> m >> x;
    ll l = 0, r = min(c, m);
    while (l < r) {
      ll mid = (l + r + 1) / 2;
      ll det = c + m + x - 2 * mid;
      if (det >= mid) {
        l = mid;
      } else {
        r = mid - 1;
      }
    }
    cout << r << endl;
  }
  return 0;
}
