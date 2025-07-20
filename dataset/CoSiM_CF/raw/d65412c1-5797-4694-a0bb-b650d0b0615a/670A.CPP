#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
using namespace std;
int main() {
  long long min = 0, n, c = 0, p;
  cin >> n;
  c += (n / 7) * 2;
  min = c;
  if (n % 7 == 1)
    c++;
  else if (n % 7 >= 2 && n % 7 < 7)
    c += 2;
  if (n % 7 == 6)
    min++;
  cout << min << ' ' << c << "\n";
}