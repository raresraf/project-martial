#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w, cost = 0;
  cin >> k >> n >> w;
  int num = w, i = 1;

  while (num > 0) {
    cost += i * k;
    i++;
    num--;
  }

  int borrow = cost - n;
  if (cost <= n)
    cout << 0;
  else
    cout << borrow;
}
