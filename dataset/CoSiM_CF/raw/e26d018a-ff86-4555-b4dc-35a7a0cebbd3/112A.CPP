#include <iostream>
using namespace std;
int main() {

  string a, b;
  cin >> a >> b;
  int i = 0;
  while (toupper(a[i]) == toupper(b[i]) && i < a.length()) {
    i++;
  }

  if (i == a.length()) {
    cout << "0";
  } else if (toupper(a[i]) > toupper(b[i])) {
    cout << "1";
  } else {
    cout << "-1";
  }

  return 0;
}
