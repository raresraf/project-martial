#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

using namespace std;
typedef long long LL;
typedef pair<int, int> pii;

#define pb push_back
#define mp make_pair
#define sz size()
#define ln length()
#define forr(i, a, b) for (int i = a; i < b; i++)
#define rep(i, n) forr(i, 0, n)
#define all(v) v.begin(), v.end()
#define uniq(v)                                                                \
  sort(all(v));                                                                \
  v.erase(unique(all(v)), v.end())
#define clr(a) memset(a, 0, sizeof a)

#define debug if (1)
#define debugoff if (0)

#define print(x) cerr << x << " ";
#define pn() cerr << endl;
#define trace1(x) cerr << #x << ": " << x << endl;
#define trace2(x, y)                                                           \
  cerr << #x << ": " << x << " | " << #y << ": " << y << endl;
#define trace3(x, y, z)                                                        \
  cerr << #x << ": " << x << " | " << #y << ": " << y << " | " << #z << ": "   \
       << z << endl;

#define MAX 100010
#define MOD 1000000007

int main() {
  ios::sync_with_stdio(false);
  int t, n, ans = 0, a;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int k = 3;
    int cnt = 0;
    while (k--) {
      cin >> a;
      if (a == 1)
        cnt++;
    }
    if (cnt >= 2)
      ans++;
  }
  cout << ans << endl;
  return 0;
}
