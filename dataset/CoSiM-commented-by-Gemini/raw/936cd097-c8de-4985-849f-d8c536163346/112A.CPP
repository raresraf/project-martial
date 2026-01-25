// 112A - Petya and Strings

#include <iostream>

using namespace std;

int main() {
  string a, b;
  int i, l;

  cin >> a >> b;

  for (i = 0; i < a.size(); i++) {
    a[i] = char(tolower(a[i]));
    b[i] = char(tolower(b[i]));
  }

  l = a.compare(b);

  if (l < 0)
    cout << "-1";
  else if (l > 0)
    cout << 1;
  else
    cout << 0;

  return 0;
}
