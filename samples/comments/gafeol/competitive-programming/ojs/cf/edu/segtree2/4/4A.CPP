#include "bits/stdc++.h"
using namespace std;
#define fst first
#define snd second
typedef unsigned long long ull;
typedef long long ll;
typedef pair<int, int> pii;
#define eb emplace_back
#define pb push_back
#define for_tests(t, tt)                                                       \
  int t;                                                                       \
  scanf("%d", &t);                                                             \
  for (int tt = 1; tt <= t; tt++)
#ifndef ONLINE_JUDGE
#define debug(args...) fprintf(stderr, args)
#else
#define debug(args...)
#endif // ONLINE_JUDGE
template <typename T> inline T abs(T t) { return t < 0 ? -t : t; }
const ll modn = 1000000007;
inline ll mod(ll x) { return x % modn; }
#define IOS() ios::sync_with_stdio(0), cin.tie(0)

const int MAXN = 212345;

int n, m, k;
int s[MAXN];

struct arv {
  ll s, ass, add;
  bool lzass;
} tree[MAXN * 4];

void unlaze(int u, int i, int j) {
  if (tree[u].lzass) {
    ll v = tree[u].ass;
    tree[u].lzass = 0;
    tree[u].ass = 0;
    tree[u].s = (j - i + 1) * v;
    if (i != j) {
      tree[u * 2].lzass = 1;
      tree[u * 2].ass = v;
      tree[u * 2].add = 0;
      tree[u * 2 + 1].lzass = 1;
      tree[u * 2 + 1].ass = v;
      tree[u * 2 + 1].add = 0;
    }
  }
  if (tree[u].add) {
    ll v = tree[u].add;
    tree[u].add = 0;
    tree[u].s += (j - i + 1) * v;
    if (i != j) {
      tree[u * 2].add += v;
      tree[u * 2 + 1].add += v;
    }
  }
}

ll qry(int idx, int i, int j, int l, int r) {
  if (i > r || j < l)
    return 0;
  unlaze(idx, i, j);
  if (i >= l && j <= r)
    return tree[idx].s;
  int m = (i + j) / 2;
  return qry(idx * 2, i, m, l, r) + qry(idx * 2 + 1, m + 1, j, l, r);
}

void upd(int idx, int i, int j, int l, int r, int ass, int add) {
  unlaze(idx, i, j);
  if (i > r || j < l)
    return;
  if (i >= l && j <= r) {
    if (add != 0) {
      tree[idx].add += add;
    } else {
      tree[idx].add = 0;
      tree[idx].ass = ass;
      tree[idx].lzass = 1;
    }
    unlaze(idx, i, j);
    return;
  }
  int m = (i + j) / 2;
  upd(idx * 2, i, m, l, r, ass, add);
  upd(idx * 2 + 1, m + 1, j, l, r, ass, add);

  tree[idx].s = tree[idx * 2].s + tree[idx * 2 + 1].s;
}

int main() {
  IOS();
  cin >> n >> m;
  for (int a = 0; a < m; a++) {
    int t, l, r;
    cin >> t >> l >> r;
    r--;
    if (t == 1) {
      int v;
      cin >> v;
      upd(1, 0, n - 1, l, r, v, 0);
    } else if (t == 2) {
      int v;
      cin >> v;
      if (!v)
        continue;
      upd(1, 0, n - 1, l, r, 0, v);
    } else {
      cout << qry(1, 0, n - 1, l, r) << endl;
    }
  }
  return 0;
}
