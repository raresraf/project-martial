/******************************************
 *  Author : Debopriya Deb Roy
 *  Created On : Wed Jul 01 2020
 *  File : c++ file
 *******************************************/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long i = 0, j, k, l, m, n;
  cin >> n;

  while (n != 0) {
    if (n % 10 == 7 || n % 10 == 4) {
      i += 1;
    }
    n /= 10;
  }
  if (i % 10 == 7 || i % 10 == 4) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}