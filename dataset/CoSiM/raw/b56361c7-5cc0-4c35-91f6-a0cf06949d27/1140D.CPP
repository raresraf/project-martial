#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define eps 1e-7
#define M 20005
#define N 150
#define SZ 200000002
using namespace std;
typedef long long ll;
const int inf = 1000000000;
ll mod = 1000000007LL;
ll mod2 = 998244353LL;
int n, k;
int main() {
  cin >> n;
  ll ans = 0LL;
  for (int i = 2; i < n; ++i) {
    ans += (ll)i * (ll)(i + 1);
  }
  cout << ans << endl;
  return 0;
}
