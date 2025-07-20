#include <bits/stdc++.h>
#include <iostream>
#define ll long long
using namespace std;

int main() {
  vector<vector<int>> v(25, vector<int>(5, 0));
  int x, y;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> v[i][j];
      if (v[i][j]) {
        x = i;
        y = j;
      }
    }
  }
  cout << abs(2 - x) + abs(2 - y) << endl;
  return 0;
}