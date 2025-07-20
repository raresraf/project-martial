#include <bits/stdc++.h>

using namespace std;

void task() {

  int m, n;
  cin >> m >> n;

  if (n % 2 == 0) {
    cout << m * (n / 2) << '\n';
  } else {
    cout << m * (n / 2) + m / 2 << '\n';
  }
}

int main() {

  task();

  return 0;
}