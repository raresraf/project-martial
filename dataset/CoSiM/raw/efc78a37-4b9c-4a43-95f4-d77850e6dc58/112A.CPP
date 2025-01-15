#include <bits/stdc++.h>
using namespace std;
int main() {
  string a, b;
  int count = 0;
  cin >> a >> b;
  for (int i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    if (a[i] > b[i]) {
      cout << "1" << endl;
      break;
    } else if (a[i] < b[i]) {
      cout << "-1" << endl;
      break;
    } else {
      count++;
    }
  }
  if (count == a.size())
    cout << "0" << endl;
  return 0;
}