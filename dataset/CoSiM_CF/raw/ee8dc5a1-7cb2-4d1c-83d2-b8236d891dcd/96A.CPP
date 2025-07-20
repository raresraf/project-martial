#include <iostream>
using namespace std;
int main() {
  char a[101];
  int i, c = 0, d = 0;
  cin >> a;
  for (i = 0; a[i] != '\0'; i++) {
    if (a[i] == '1') {
      c++;
      d = 0;
      if (c == 7) {
        cout << "YES";
        break;
      }
    } else {
      d++;
      c = 0;
      if (d == 7) {
        cout << "YES";
        break;
      }
    }
  }
  if (c != 7 && d != 7)
    cout << "NO";
  return 0;
}
