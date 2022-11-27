#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  int w;
  cin >> w;

  if (w % 2 != 0 || w == 2)
    cout << "NO";
  else
    cout << "YES";

  return 0;
}