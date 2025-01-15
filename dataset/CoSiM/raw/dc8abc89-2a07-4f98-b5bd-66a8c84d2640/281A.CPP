#include <iostream>
using namespace std;
int main() {
  char a[1000];
  cin >> a;
  if (a[0] >= 'a' && a[0] <= 'z') {
    a[0] = a[0] + 'A' - 'a';
    cout << a << endl;
  } else
    cout << a << endl;

  return 0;
}
