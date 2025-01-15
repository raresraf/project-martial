#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int n, m, i = 0, j;
  cin >> n;
  while (n != 0) {
    m = n % 10;
    n = n / 10;
    if (m == 4 || m == 7)
      i++;
  }

  if (i == 4 || i == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
