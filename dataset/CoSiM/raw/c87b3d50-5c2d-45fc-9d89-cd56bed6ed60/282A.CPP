#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int x, y = 0;
  cin >> x;
  string c[x];
  for (int i = 0; i < x; i++) {
    cin >> c[i];

    if (c[i] == "X++") {
      y++;
    } else if (c[i] == "++X") {
      ++y;
    } else if (c[i] == "X--") {
      y--;
    } else if (c[i] == "--X") {
      --y;
    }
  }
  cout << y << endl;
}