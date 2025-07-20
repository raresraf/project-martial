#include <bits/stdc++.h>

using namespace std;

int main() {
  int ind, ban, fbc, total = 0;
  cin >> fbc;
  cin >> ind;
  cin >> ban;
  for (int i = 1; i <= ban; i++) {
    total += i * fbc;
  }
  int wad = total - ind;
  if (wad < 0)
    wad = 0;
  cout << wad << endl;

  return 0;
}
