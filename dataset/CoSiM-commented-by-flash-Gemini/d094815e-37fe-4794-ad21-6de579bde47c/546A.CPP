#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int k, n, w;
  cin >> k >> n >> w;
  int price = 0;
  for (int i = 1; i <= w; i++) {
    price += (k * i);
  }
  if (price - n > 0)
    cout << price - n;
  else
    cout << "0";
  return 0;
}