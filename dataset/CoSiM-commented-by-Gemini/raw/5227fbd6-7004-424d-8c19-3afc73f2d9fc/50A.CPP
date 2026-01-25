#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, m;
  cin >> n >> m;

  int total = 0;

  if (n % 2 == 0) {
    total = (n / 2) * m;
  } else if (m % 2 == 0) {
    total = (m / 2) * n;
  } else {
    total = ((n / 2) * m) + (m / 2);
  }

  cout << total << endl;

  return 0;
}