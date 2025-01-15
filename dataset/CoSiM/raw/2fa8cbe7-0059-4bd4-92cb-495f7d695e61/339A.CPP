#include <iostream>
using namespace std;

int main() {
  string a;
  int i, ct1 = 0, ct2 = 0, ct3 = 0;
  cin >> a;
  for (i = 0; a[i] != '\0'; i += 2) {
    if (a[i] == '1')
      ct1 += 2;
    else if (a[i] == '2')
      ct2 += 2;
    else if (a[i] == '3')
      ct3 += 2;
  }
  for (i = 0; i < ct1; i = i + 2)
    a[i] = '1';
  for (; i < (ct1 + ct2); i = i + 2)
    a[i] = '2';
  for (; i < (ct3 + ct2 + ct1); i = i + 2)
    a[i] = '3';
  cout << a;
  return 0;
}
