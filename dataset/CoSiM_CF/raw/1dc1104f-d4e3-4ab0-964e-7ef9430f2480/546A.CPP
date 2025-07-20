#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, w;
  long long int n;
  int cost = 0;
  long long int borrow = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    cost = cost + k * i;
  }
  if (n < cost) {
    borrow = cost - n;
    cout << borrow << endl;
  } else {
    cout << borrow << endl;
  }
  return 0;
}
