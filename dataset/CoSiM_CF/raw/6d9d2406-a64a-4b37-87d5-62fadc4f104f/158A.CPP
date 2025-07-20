#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  int n, k;
  cin >> n >> k;
  int s = 0;

  while (n--) {
    int a;
    cin >> a;
    if (a <= k)
      break;
    else
      s++;
  }

  cout << s << endl;
  return 0;
}