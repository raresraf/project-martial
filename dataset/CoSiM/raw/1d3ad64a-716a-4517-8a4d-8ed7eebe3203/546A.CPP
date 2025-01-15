#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n, w, c = 0;
  cin >> k >> n >> w;
  for (int i = 0; i < w; i++)
    c += (i + 1) * k;
  if (c <= n)
    cout << 0;
  else
    cout << c - n;
  return 0;
}
