#include <algorithm>
#include <iostream>
#include <string.h>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  int x = 0, y = x, r = 0;
  string a;
  cin >> a;
  bool result = 0;
  for (int i = 0; i < a.length(); i++) {
    x = count(a.begin() + i, a.begin() + i + 7, '1');
    y = count(a.begin() + i, a.begin() + i + 7, '0');
    if (x >= 7 || y >= 7)
      result = 1;
  }
  if (result) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
