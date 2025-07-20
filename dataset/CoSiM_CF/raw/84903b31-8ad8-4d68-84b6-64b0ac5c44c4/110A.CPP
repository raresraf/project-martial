#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  long long int x, count = 0;
  cin >> x;

  while (x) {
    if (x % 10 == 4 || x % 10 == 7)
      count++;

    x = x / 10;
  }

  if (count == 7 || count == 4)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}