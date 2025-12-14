#include <bits/stdc++.h>
using namespace std;
int sum = 0, n, a[111111], k = 0, total = 0;
int main() {
  cin >> n;
  for (int i = 0; i < n; i++)
    cin >> a[i], total += a[i];

  sort(a, a + n, greater<int>());

  for (int i = 0; i < n; i++)
    if (sum > total - sum)
      break;
    else
      sum += a[i], k++;
  cout << k;
}
