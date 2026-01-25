#include <bits/stdc++.h>
#define OPTIMASI                                                               \
  cin.tie(0);                                                                  \
  cout.tie(0);                                                                 \
  ios_base::sync_with_stdio(0);
typedef long long int lli;
using namespace std;

int main() {
  OPTIMASI

  long long int n, k;
  cin >> n;

  k = n / 7;
  n = n - k * 7;

  if (n == 6)
    cout << k * 2 + 1 << " ";
  else
    cout << k * 2 << " ";
  if (n > 2)
    cout << k * 2 + 2;
  else {
    cout << k * 2 + n;
  }
  cout << endl;

  return 0;
}