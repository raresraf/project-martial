/*
 *	Copyright (C) Ds ds303077135@gmail.com
 *	File Name     : A.cpp
 *	Creation Time : 2013/01/08 18:34:10
 *	Environment   : OS X 10.8
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#define rep(i, n) for (int i = 1; i <= n; ++i)
using namespace std;

int m, n, q, r = 100001, a[100010];
long long tot;

int main() {
  scanf("%d", &m);
  rep(i, m) scanf("%d", &q), r = min(r, q);
  scanf("%d", &n);
  rep(i, n) scanf("%d", &a[i]), tot += a[i];
  sort(a + 1, a + 1 + n);
  int pos = n;
  while (pos >= 1) {
    if (pos > r) {
      tot -= a[pos - r];
      if (pos - r > 1)
        tot -= a[pos - r - 1], pos = pos - r - 2;
      else
        pos = pos - r - 1;
    } else
      break;
  }
  cout << tot << endl;
  return 0;
}
