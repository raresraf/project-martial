#include <iostream>
using namespace std;
int main() {

  char a[20];
  cin >> a;
  int i, flag, l, c4, c7, c;
  l = 0;
  c4 = 0;
  c7 = 0;
  c = 0;
  while (a[l] != '\0')
    l++;
  for (i = 0; i < l; i++) {
    if (a[i] == '4')
      c4++;
    else if (a[i] == '7')
      c7++;
  }
  if (c4 + c7 == 4 || c4 + c7 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
