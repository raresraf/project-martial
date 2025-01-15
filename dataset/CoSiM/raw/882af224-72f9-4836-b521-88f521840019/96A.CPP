#include <iostream>
using namespace std;

int main() {
  string a;
  int i, m = 0, ct = 1, b;
  cin >> a;
  b = a[0];
  for (i = 1; a[i] != '\0'; i++) {
    if (a[i] == b)
      ct++;
    else {
      if (ct > m)
        m = ct;
      ct = 1;
      b = a[i];
    }
  }
  if (ct > m)
    m = ct;
  if (m >= 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
