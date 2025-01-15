#include <iostream>
using namespace std;
int strlength(char a[]) {
  int count = 0;
  while (a[count] != '\0')
    count++;
  return count;
}
int main() {
  char a[100];
  cin >> a;
  //	cout << a << strlength(a) << endl;
  char b[100];
  cin >> b;
  //	cout << b << a << endl;
  int l, i, ans;
  l = strlength(a);
  for (i = 0; i < l; i++) {
    // cout << "check " << i << endl;
    if (a[i] >= 'A' && a[i] <= 'Z')
      a[i] = a[i] - 'A' + 'a';
    else
      a[i] = a[i];
    if (b[i] >= 'A' && b[i] <= 'Z')
      b[i] = b[i] - 'A' + 'a';
    else
      b[i] = b[i];
    if (a[i] > b[i]) {
      ans = 1;
      break;
    } else if (b[i] > a[i]) {
      ans = -1;
      break;
    } else
      ans = 0;
  }
  cout << ans << endl;
  return 0;
}
