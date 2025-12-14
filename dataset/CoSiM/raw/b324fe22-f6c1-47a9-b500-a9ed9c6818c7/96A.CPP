#include <iostream>
#include <string>
using namespace std;

int main() {
  string a;
  int flag1 = 0, flag2 = 0, c1 = 1, c2 = 0;
  cin >> a;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == a[i + 1]) {
      c1++;
      if (c1 == 7)
        break;
    } else
      c1 = 1;
  }
  if (c1 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
