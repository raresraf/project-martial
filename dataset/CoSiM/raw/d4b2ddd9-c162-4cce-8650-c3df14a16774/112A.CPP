// 112A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] >= 65 && a[i] <= 90) {
      a[i] += 32;
    }
    if (b[i] >= 65 && b[i] <= 90) {
      b[i] += 32;
    }
  }
  if (a < b)
    cout << "-1";
  else if (a > b)
    cout << "1";
  else if (a == b)
    cout << "0";
  return 0;
}
