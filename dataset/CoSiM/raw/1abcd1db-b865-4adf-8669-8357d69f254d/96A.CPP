#include <iostream>
#include <string.h>
using namespace std;

int main() {
  int t = 0, p = 0;
  string a;
  cin >> a;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == '0') {
      t++;
      p = 0;
    } else if (a[i] == '1') {
      p++;
      t = 0;
    } else
      cout << "";
    if ((t > 6) || (p > 6)) {
      cout << "YES" << endl;
      break;
    }
  }
  if ((t < 6) && (p < 6))
    cout << "NO" << endl;
}
