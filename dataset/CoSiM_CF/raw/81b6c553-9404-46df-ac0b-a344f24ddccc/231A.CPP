// coding team codeforces

#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a, b, c;
  int co = 0;
  while (n--) {
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      co++;
  }

  cout << co << endl;
}
