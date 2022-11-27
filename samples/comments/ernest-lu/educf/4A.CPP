#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef pair<int, int> pii;
#define mp make_pair
#define pb push_back
#define f first
#define s second

// var
int n, m, a[100001], od[100001], ev[100001];

void upo(int g, int k) {
  for (; g <= n; g += (g & -g))
    od[g] += k;
}

void upe(int g, int k) {
  for (; g <= n; g += (g & -g))
    ev[g] += k;
}

LL ge(int g) {
  LL res = 0;
  for (; g > 0; g -= (g & -g))
    res += ev[g];
  return res;
}

LL go(int g) {
  LL res = 0;
  for (; g > 0; g -= (g & -g))
    res += od[g];
  return res;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
    if (i % 2)
      upo(i, a[i]);
    else
      upe(i, a[i]);
  }
  cin >> m;
  while (m--) {
    int tt;
    cin >> tt;
    if (tt) {
      int l, r;
      cin >> l >> r;
      if (l % 2)
        cout << (go(r) - go(l - 1)) - (ge(r) - ge(l - 1)) << endl;
      else
        cout << (ge(r) - ge(l - 1)) - (go(r) - go(l - 1)) << endl;
      continue;
    }
    int g, k;
    cin >> g >> k;
    if (g % 2) {
      upo(g, k - a[g]);
      a[g] = k;
    } else {
      upe(g, k - a[g]);
      a[g] = k;
    }
  }

  cout.flush();
  return 0;
}