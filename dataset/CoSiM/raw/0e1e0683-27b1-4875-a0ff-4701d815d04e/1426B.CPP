#include <algorithm>
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
#define N 300005
using namespace std;
typedef long long ll;
ll mod = 1000000007LL;
ll mod2 = 998244353LL;
int t, n, m;
vector<pair<pair<int, int>, pair<int, int>>> tiles;
int main() {
  cin >> t;
  for (int cas = 1; cas <= t; ++cas) {
    scanf("%d%d", &n, &m);
    bool found1 = false;
    bool found2 = false;
    tiles.clear();
    for (int i = 1; i <= n; ++i) {
      int a, b, c, d;
      scanf("%d%d%d%d", &a, &b, &c, &d);
      tiles.push_back(make_pair(make_pair(a, b), make_pair(c, d)));
      if (b == c) {
        found2 = true;
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int a1 = tiles[i].first.first;
        int b1 = tiles[i].first.second;
        int c1 = tiles[i].second.first;
        int d1 = tiles[i].second.second;
        int a2 = tiles[j].first.first;
        int b2 = tiles[j].first.second;
        int c2 = tiles[j].second.first;
        int d2 = tiles[j].second.second;
        if (c1 == b2 && b1 == c2 && a1 == a2 && d1 == d2) {
          found1 = true;
          break;
        }
      }
      if (found1)
        break;
    }
    if (m % 2) {
      cout << "NO" << endl;
    } else {
      if (m <= 2) {
        if (found2)
          cout << "YES" << endl;
        else
          cout << "NO" << endl;
      } else {
        if (found1 && found2)
          cout << "YES" << endl;
        else
          cout << "NO" << endl;
      }
    }
  }
  return 0;
}
