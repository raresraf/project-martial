#include <bits/stdc++.h>
#include <iostream>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  int n, k, c = 0;
  cin >> n;
  int a[n];
  cin >> k;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  int val = a[k - 1];

  for (int i = 0; i < n; i++) {
    if (a[i] >= val && a[i] > 0) {
      c++;
    }
  }
  cout << c << endl;
  return 0;
}
