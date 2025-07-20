#include <iostream>
using namespace std;
int main() {
  char a[60];
  int i, b, c = 0;
  cin >> b;
  for (i = 1; i <= b; i++) {
    cin >> a[i];
  }
  for (i = 0; i < b; i++) {
    if (a[i] == a[i + 1]) {
      c++;
    }
  }
  cout << c << endl;
}
