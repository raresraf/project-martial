#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  int n, maxd, mind;
  cin >> n;
  mind = (n / 7) * 2;
  maxd = (n / 7) * 2;
  if (n % 7 > 5)
    mind++;
  if (n % 7 >= 2)
    maxd += 2;
  else if (n % 7 == 1)
    maxd += 1;
  cout << mind << " " << maxd;
  return 0;
}