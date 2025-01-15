#include <bits/stdc++.h>
using namespace std;
int main() {
  int sum1 = 0, sum2 = 0, cont = 0, diff;
  int n, aa[105];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> aa[i];
    sum1 += aa[i];
  }
  sort(aa, aa + n);
  for (int i = n - 1; i >= 0; i--) {
    sum2 += aa[i];
    diff = sum1 - sum2;
    cont++;
    if (sum2 > diff)
      break;
  }
  cout << cont << endl;
}
