#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, n, w;
  int total = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++)
    total += i * k;
  if (total < n)
    cout << 0;
  else
    cout << total - n;
  return 0;
}
