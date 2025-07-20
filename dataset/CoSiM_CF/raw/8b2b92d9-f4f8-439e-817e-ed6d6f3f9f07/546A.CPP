#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w, totalCost = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++)
    totalCost += (i * k);
  if (totalCost > n)
    cout << totalCost - n;
  else
    cout << 0;
}
