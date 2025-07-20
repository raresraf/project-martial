#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int i;
  bool OK = false;
  string a;
  cin >> a;
  i = 0;
  if (a.length() >= 7) {
    while (i < a.length() - 6) {
      if (a[i] == a[i + 1] && a[i] == a[i + 2] && a[i] == a[i + 3] &&
          a[i] == a[i + 4] && a[i] == a[i + 5] && a[i] == a[i + 6]) {
        OK = true;
        break;
      }
      i++;
    }
  }
  if (OK)
    cout << "YES";
  else
    cout << "NO";
}
