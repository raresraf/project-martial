#include <bits/stdc++.h>

using namespace std;
int main(int argc, char const *argv[]) {
  int k, n, w;
  cin >> k;
  cin >> n;
  cin >> w;

  int money = 0;
  int amt = 1;

  for (int i = 1; i <= w; i++) {
    amt = k * i;
    money = money + amt;
  }

  if (money > n) {
    int x = money - n;
    cout << x;
  }

  else {
    cout << "0";
  }

  return 0;
}