#include <iostream>
#include <string>
using namespace std;
int main() {
  string a, b = "hello";
  cin >> a;
  int i, j = 0, cont = 0;
  for (i = 0; i < a.length(); i++) {
    if (a[i] == b[j]) {
      j++;
      cont++;
    }
    if (cont == 5)
      break;
  }
  if (cont == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}