#include <bits/stdc++.h>
#include <iostream>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  if (a == b)
    cout << "0" << endl;
  else {
    for (int i = 0; i < a.size(); i++) {
      if (a[i] < b[i]) {
        cout << "-1" << endl;
        break;
      }
      if (a[i] > b[i]) {
        cout << "1" << endl;
        break;
      }
    }
  }
  return 0;
}
