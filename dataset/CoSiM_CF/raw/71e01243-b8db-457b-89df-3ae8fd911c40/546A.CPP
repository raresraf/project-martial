#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int price = 0;

  for (int i = 1; i <= w; ++i) {
    price += i * k;
  }

  int finalprice = price - n;

  if (finalprice > 0) {
    cout << finalprice;
  }
  if (finalprice == 0 || finalprice < 0) {
    cout << 0;
  }
}
