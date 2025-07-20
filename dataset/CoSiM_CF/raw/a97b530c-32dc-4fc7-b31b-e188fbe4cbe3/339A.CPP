#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  string a;
  cin >> a;
  int r = (a.length() / 2) + 1;
  // cout<<r;
  int ar[r], j = 0;
  for (int i = 0; i < a.length(); i += 2) {
    ar[j] = a.at(i) - '0';
    j++;
  }
  sort(ar, ar + r);
  for (int i = 0; i < r; i++) {
    if (i + 1 != r) {
      cout << ar[i] << "+";
    } else {
      cout << ar[i];
    }
  }

  return 0;
}
