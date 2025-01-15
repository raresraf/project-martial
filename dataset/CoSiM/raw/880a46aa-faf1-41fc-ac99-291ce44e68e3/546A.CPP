/// Problem Name: Soldier and Bananas
/// Problem Link: https://codeforces.com/problemset/problem/546/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  long long k, n, w, d = 0;
  scanf("%lld %lld %lld", &k, &n, &w);
  for (int i = 0; i < w; i++)
    d += k * (i + 1);
  d = d - n;
  if (d < 0)
    d = 0;
  printf("%lld\n", d);
  return 0;
}
