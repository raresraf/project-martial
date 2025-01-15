#include <bits/stdc++.h>
using namespace std;

int main() {
  int m, n;
  cin >> m >> n;
  int temp = floor(m / 2);
  if (n % 2 == 0 && n != 1 && m != 1)
    cout << m * (n / 2);
  else if (n == 1 || m == 1)
    cout << m * floor(n / 2);
  else
    cout << m * ((n - 1) / 2) + temp;
}
