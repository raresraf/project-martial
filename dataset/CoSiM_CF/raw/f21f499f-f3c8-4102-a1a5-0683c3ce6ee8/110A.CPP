#include <iostream>
using namespace std;
int main() {
  string a;
  cin >> a;
  int c = 0, b = 0;
  for (int i = 0; i <= a.size(); i++) {
    if (a[i] == '4') {
      c++;
    }
    if (a[i] == '7') {
      b++;
    }
  }
  if (b + c == 0) {
    cout << "NO";
  } else if ((b + c) == 4 || (b + c) == 7) {
    cout << "YES";
  } else
    cout << "NO";
}
