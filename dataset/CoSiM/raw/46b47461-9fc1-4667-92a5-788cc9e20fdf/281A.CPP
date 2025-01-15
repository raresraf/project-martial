#include <iostream>
#include <string>
using namespace std;
int main() {
  string a;
  cin >> a;
  char temp = a[0];
  if (temp <= 'Z' && temp >= 'A') {
    cout << a;
  } else {
    a[0] -= 'z' - 'Z';
    cout << a;
  }
  return 0;
}
