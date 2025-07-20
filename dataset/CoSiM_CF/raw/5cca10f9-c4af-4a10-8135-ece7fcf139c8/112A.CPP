/*
Link: https://codeforces.com/problemset/problem/112/A
@uthor: chavodon
*/

#include <iostream>
#include <string>

using namespace std;

int main() {
  string a, b;
  int i, flag = 0;
  cin >> a;
  cin >> b;
  for (i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    if (a[i] < b[i]) {
      cout << "-1";
      flag = 1;
      break;
    } else if (a[i] > b[i]) {
      cout << "1";
      flag = 1;
      break;
    }
  }
  if (flag == 0)
    cout << "0";
  return 0;
}