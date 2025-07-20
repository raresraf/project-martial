#include <bits/stdc++.h>
using namespace std;
int main(void) {
  int k, n, w, sum = 0;
  scanf("%d%d%d", &k, &n, &w);
  for (int i = 1; i <= w; i++)
    sum += (k * i);
  if (sum <= n)
    cout << 0 << endl;
  else
    cout << sum - n << endl;
}