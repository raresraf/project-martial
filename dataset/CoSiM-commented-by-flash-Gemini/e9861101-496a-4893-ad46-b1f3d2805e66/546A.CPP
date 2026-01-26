#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int ris = k * (w) * (w + 1) / 2 - n;
  if (ris > 0)
    cout << ris << "\n";
  else
    cout << 0 << "\n";
  return 0;
}
