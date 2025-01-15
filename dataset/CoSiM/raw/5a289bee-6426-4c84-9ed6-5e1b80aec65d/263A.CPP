#include <bits/stdc++.h>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int a[25];
  int k;
  for (int i = 0; i < 25; i++) {
    cin >> a[i];
    if (a[i] == 1)
      k = i + 1;
  }
  if ((k == 1) || (k == 5) || (k == 21) || (k == 25))
    cout << 4 << endl;
  if ((k == 4) || (k == 10) || (k == 24) || (k == 20) || (k == 22) ||
      (k == 16) || (k == 6) || (k == 2))
    cout << 3 << endl;
  if ((k == 3) || (k == 7) || (k == 11) || (k == 17) || (k == 23) ||
      (k == 19) || (k == 15) || (k == 9))
    cout << 2 << endl;
  if ((k == 8) || (k == 12) || (k == 14) || (k == 18))
    cout << 1 << endl;
  if (k == 13)
    cout << 0;
  return 0;
}