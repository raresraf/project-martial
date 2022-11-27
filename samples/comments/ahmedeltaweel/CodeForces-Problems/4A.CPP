#include <bits/stdc++.h>

using namespace std;

int main(void) {

  // freopen("c.in" , "r" , stdin);

  int n;
  cin >> n;

  if (n % 2 == 0 && n != 2) {
    cout << "YES" << endl;
    return 0;
  }

  cout << "NO" << endl;
  return 0;
}
